use crate::error::{DecodeError, Error, Result};
use crate::fileformat::code_object::CodeObjectV3;
use bitflags::bitflags;
use goblin::elf::{sym, Elf};
use goblin::Object;
use std::cmp::min;
use std::collections::HashMap;
use std::io;
use std::io::ErrorKind;
use std::slice::from_raw_parts;
use std::str::FromStr;

// Kernel descriptor according to https://llvm.org/docs/AMDGPUUsage.html#code-object-v3-kernel-descriptor
#[repr(C)]
struct KernelDescriptor {
    group_segment_fixed_size: u32,
    private_segment_fixed_size: u32,
    kern_arg_size: u32,
    _reserved0: u32,
    kernel_code_entry_byte_offset: i64,
    _reserved1: [u32; 5],
    pgmrsrcs3: u32,
    pgmrsrcs1: u32,
    pgmrsrcs2: u32,
    // 16 bits of setup
    setup: u32,
    _reserved2: u32,
}

static_assertions::const_assert_eq!(std::mem::size_of::<KernelDescriptor>(), 64);

bitflags! {
    #[derive(Default)]
    pub struct SGPRSetup : u8 {
        const SGPR_PRIVATE_SEGMENT_BUFFER = 1;
        const SGPR_DISPATCH_PTR = 1 << 1;
        const SGPR_QUEUE_PTR = 1 << 2;
        const SGPR_KERNARG_SEGMENT_PTR = 1 << 3;
        const SGPR_DISPATCH_ID = 1 << 4;
        const SGPR_FLAT_SCRATCH_INIT = 1 << 5;
        const SGPR_PRIVATE_SEGMENT_SIZE = 1 << 6;
    }
}

#[derive(Copy, Clone)]
pub struct PGMRSRC2(u32);

impl PGMRSRC2 {
    const ENABLE_VGPR_WORKITEM_ID_MASK: u32 = 3 << 11;
    const ENABLE_VGPR_WORKITEM_ID_OFFSET: u32 = 11;
    pub const SGPR_PRIVATE_SEGMENT: Self = PGMRSRC2(1);
    pub const SGPR_WORKGROUP_ID_X: Self = PGMRSRC2(1 << 7);
    pub const SGPR_WORKGROUP_ID_Y: Self = PGMRSRC2(1 << 8);
    pub const SGPR_WORKGROUP_ID_Z: Self = PGMRSRC2(1 << 9);
    pub const SGPR_WORKGROUP_INFO: Self = PGMRSRC2(1 << 10);

    pub fn from(v: u32) -> Self {
        Self(v)
    }
    pub fn enable_vgpr_workitem_id(&self) -> Result<u8> {
        let v =
            (self.0 & Self::ENABLE_VGPR_WORKITEM_ID_MASK) >> Self::ENABLE_VGPR_WORKITEM_ID_OFFSET;
        if v < 3 {
            Ok(v as u8)
        } else {
            Err(Error::DecodeError(DecodeError::InvalidDescriptor))
        }
    }

    pub fn contains(&self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ArgInfo<'a> {
    pub name: Option<&'a str>,
    pub offset: usize,
    pub length: usize,
    pub value_kind: KernelArgValueKind,
}

// Kernel Argument Kind according to https://llvm.org/docs/AMDGPUUsage.html#amdgpu-amdhsa-code-object-kernel-argument-metadata-map-table-v3
#[derive(Debug, Copy, Clone)]
pub enum KernelArgValueKind {
    ByValue,
    GlobalBuffer,
    DynamicSharedPointer,
    Sampler,
    Image,
    Pipe,
    Queue,
    HiddenGlobalOffsetX,
    HiddenGlobalOffsetY,
    HiddenGlobalOffsetZ,
    HiddenNone,
    HiddenPrintfBuffer,
    HiddenDefaultQueue,
    HiddenCompletionAction,
    HiddenMultiGridSyncArg,
}

impl FromStr for KernelArgValueKind {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "by_value" => Self::ByValue,
            "global_buffer" => Self::GlobalBuffer,
            "dynamic_shared_pointer" => Self::DynamicSharedPointer,
            "sampler" => Self::Sampler,
            "image" => Self::Image,
            "pipe" => Self::Pipe,
            "queue" => Self::Queue,
            "hidden_global_offset_x" => Self::HiddenGlobalOffsetX,
            "hidden_global_offset_y" => Self::HiddenGlobalOffsetY,
            "hidden_global_offset_z" => Self::HiddenGlobalOffsetZ,
            "hidden_none" => Self::HiddenNone,
            "hidden_printf_buffer" => Self::HiddenPrintfBuffer,
            "hidden_default_queue" => Self::HiddenDefaultQueue,
            "hidden_completion_action" => Self::HiddenCompletionAction,
            "hidden_multigrid_sync_arg" => Self::HiddenMultiGridSyncArg,
            _ => {
                return Err(Error::ELFError(goblin::error::Error::Malformed(
                    "Unrecognized kernel argument kind: ".to_string() + s,
                )));
            }
        })
    }
}

#[derive(Debug, Default, Clone)]
pub struct KernelInfo<'a> {
    pub name: &'a str,
    pub code: &'a [u32],
    // Various PGMRSRC flags for the kernel according to the LLVM User Guide for AMDGPU Backend
    pub pgmrsrcs: [u32; 3],
    pub setup: SGPRSetup,
    pub kern_arg_size: u32,
    pub kern_arg_segment_align: u32,
    pub group_segment_fixed_size: u32,
    pub private_segment_fixed_size: u32,
    pub arguments: Vec<ArgInfo<'a>>,
}

impl<'a> KernelInfo<'a> {
    const UNKNOWN_KERN_ARG_SIZE: u32 = 0;
    fn update_from_descriptor(&mut self, d: &KernelDescriptor) {
        self.pgmrsrcs = [d.pgmrsrcs1, d.pgmrsrcs2, d.pgmrsrcs3];
        self.kern_arg_size = d.kern_arg_size;
        self.group_segment_fixed_size = d.group_segment_fixed_size;
        self.private_segment_fixed_size = d.private_segment_fixed_size;
        self.setup = SGPRSetup::from_bits_truncate(d.setup as u8);
    }
}

pub struct AMDGPUELF<'a> {
    pub(crate) kernels: Vec<KernelInfo<'a>>,
}

#[derive(Ord, PartialOrd, Eq, PartialEq)]
enum SymbolType {
    Function,
    Object,
}

struct Parser<'a> {
    bytes: &'a [u8],
    elf: Elf<'a>,
    syms: HashMap<&'a str, (SymbolType, u64)>,
}

impl<'a> Parser<'a> {
    fn new(bytes: &'a [u8]) -> Result<Parser> {
        let elf = match Object::parse(bytes)? {
            Object::Elf(x) => Ok(x),
            _ => Err(Error::ELFError(goblin::error::Error::Malformed(
                "Invalid ELF".to_string(),
            ))),
        }?;

        let syms = Self::parse_sym_table(&elf);
        Ok(Parser { bytes, elf, syms })
    }

    fn parse(&mut self) -> Result<Vec<KernelInfo<'a>>> {
        let notes = self.elf.iter_note_sections(self.bytes, None);
        if notes.is_none() {
            return Ok(Vec::new());
        }

        let mut info = Vec::new();
        for x in notes.unwrap() {
            let n = x?;
            if n.name == "AMDGPU" {
                CodeObjectV3::try_foreach(n.desc, |k: &mut KernelInfo, desc_symbol| {
                    let func_not_found = |k: &KernelInfo| {
                        Error::ELFError(goblin::error::Error::Malformed(format!(
                            "function {} not found",
                            k.name
                        )))
                    };
                    let start = self
                        .syms
                        .get(desc_symbol)
                        .ok_or_else(|| func_not_found(k))?
                        .1;
                    let off = self.vma_span_to_file_offset(start, 64)?;
                    if off.1 != 64 {
                        return Err(Error::ELFError(goblin::error::Error::Malformed(format!(
                            "size of kernel descriptor of {} is expected to be 64 bytes",
                            k.name
                        ))));
                    }
                    let blob = self.get_blob(off.0 as usize, off.1 as usize)?;
                    let d = unsafe { &*(blob.as_ptr() as *const KernelDescriptor) };
                    k.update_from_descriptor(d);
                    let code_vma = if d.kernel_code_entry_byte_offset < 0 {
                        start - (-d.kernel_code_entry_byte_offset) as u64
                    } else {
                        start + d.kernel_code_entry_byte_offset as u64
                    };
                    if code_vma % 256 != 0 {
                        return Err(Error::ELFError(goblin::error::Error::Malformed(format!(
                            "code block of {} is expected to be aligned by the 256-byte boundary",
                            k.name
                        ))));
                    } else if self.syms.get(k.name).ok_or_else(|| func_not_found(k))?.1 != code_vma
                    {
                        return Err(Error::ELFError(goblin::error::Error::Malformed(format!(
                            "Unexpected code VMA for {}",
                            k.name
                        ))));
                    }

                    // update kern_arg_size from arg info
                    if k.kern_arg_size == KernelInfo::UNKNOWN_KERN_ARG_SIZE {
                        k.kern_arg_size = k
                            .arguments
                            .iter()
                            .map(|a| (a.offset + a.length) as u32)
                            .fold(0, |size, bound| size.max(bound));
                    }
                    info.push((k.clone(), code_vma));
                    Ok(())
                })?;
            }
        }

        let kernels = self.update_code_blocks(info)?;
        Ok(kernels)
    }

    fn update_code_blocks(&self, info: Vec<(KernelInfo<'a>, u64)>) -> Result<Vec<KernelInfo<'a>>> {
        let mut func_vmas: Vec<u64> = self
            .syms
            .iter()
            .filter_map(|(_, (ty, vma))| match ty {
                SymbolType::Function => Some(*vma),
                _ => None,
            })
            .collect();
        func_vmas.sort();
        if info.is_empty() || func_vmas.is_empty() {
            return Ok(Vec::new());
        }

        let text_sec = self
            .elf
            .section_headers
            .iter()
            .find(|h| {
                matches!(self.elf.shdr_strtab.get_at(h.sh_name),
                Some(x) if x == ".text")
            })
            .ok_or_else(|| {
                Error::ELFError(goblin::error::Error::Malformed(
                    "No .text section".to_string(),
                ))
            })?;
        let text_end = text_sec.sh_addr + text_sec.sh_size;
        if *func_vmas.last().unwrap() >= text_end {
            return Err(Error::ELFError(goblin::error::Error::Malformed(
                "function out of bounds".to_string(),
            )));
        }
        func_vmas.push(text_end);

        let mut r = Vec::new();
        for (mut k, vma) in info {
            let next = func_vmas.binary_search(&vma).unwrap() + 1;
            let end = func_vmas[next];
            if end % 4 != 0 {
                return Err(Error::ELFError(goblin::error::Error::Malformed(
                    "unaligned boundary for function".to_string(),
                )));
            }
            let size = end - vma;
            let slice = self.vma_span_to_file_offset(vma, size)?;
            let blob = self.get_blob(slice.0 as usize, slice.1 as usize)?;
            unsafe {
                k.code = from_raw_parts(blob.as_ptr() as *const u32, (size / 4) as usize);
            }
            r.push(k);
        }
        Ok(r)
    }

    fn get_blob(&self, offset: usize, size: usize) -> Result<&[u8]> {
        if offset + size > self.bytes.len() {
            Err(Error::IOError(io::Error::from(ErrorKind::UnexpectedEof)))
        } else {
            Ok(&self.bytes[offset..offset + size])
        }
    }

    // Find the corresponding region in the file for a VMA span.
    // The size of the actual region can be smaller or even zero if it comes from .bss
    fn vma_span_to_file_offset(&self, start: u64, size: u64) -> Result<(u64, u64)> {
        let ph = self
            .elf
            .program_headers
            .iter()
            .find(|p| p.p_vaddr <= start && start + size <= p.p_vaddr + p.p_memsz)
            .ok_or_else(|| io::Error::from(ErrorKind::NotFound))?;
        Ok((ph.p_offset + (start - ph.p_vaddr), min(size, ph.p_filesz)))
    }

    fn parse_sym_table(elf: &Elf<'a>) -> HashMap<&'a str, (SymbolType, u64)> {
        elf.syms
            .iter()
            .filter_map(|x| {
                let (ty, vis) = (x.st_type(), x.st_bind());
                match ty {
                    sym::STT_OBJECT if vis == sym::STB_GLOBAL => Some((
                        elf.strtab.get_at(x.st_name)?,
                        (SymbolType::Object, x.st_value),
                    )),
                    sym::STT_FUNC if vis == sym::STB_GLOBAL => Some((
                        elf.strtab.get_at(x.st_name)?,
                        (SymbolType::Function, x.st_value),
                    )),
                    _ => None,
                }
            })
            .collect::<HashMap<&'a str, (SymbolType, u64)>>()
    }
}

impl<'a> AMDGPUELF<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<AMDGPUELF> {
        let mut parser = Parser::new(bytes)?;
        let kernels = parser.parse()?;
        Ok(AMDGPUELF { kernels })
    }

    pub fn kernels(&self) -> &[KernelInfo<'a>] {
        self.kernels.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vgpr_workitem_id() {
        assert_eq!(PGMRSRC2(0).enable_vgpr_workitem_id().unwrap(), 0);
        assert_eq!(PGMRSRC2(1 << 11).enable_vgpr_workitem_id().unwrap(), 1);
        assert_eq!(PGMRSRC2(2 << 11).enable_vgpr_workitem_id().unwrap(), 2);
        assert!(PGMRSRC2(3 << 11).enable_vgpr_workitem_id().is_err());
    }
}
