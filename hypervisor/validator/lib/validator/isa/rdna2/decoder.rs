use crate::error::{DecodeError, Error, Result};
use crate::isa::rdna2::isa::{Opcode, Operand};
use crate::isa::rdna2::opcodes::*;
use bitfield::bitfield;
use bitflags::bitflags;
use smallvec::smallvec;
use smallvec::SmallVec;
use std::fmt::{Display, Formatter};
use std::io;
use std::io::ErrorKind;

/**
 * An abstraction for the RDNA2 instruction. Some conventions:
 *
 * Operand 0 is always the destination registers (Marked as reserved if it is a store instruction)
 **/
#[derive(Clone, Debug)]
pub struct Instruction {
    pub(crate) op: Opcode,
    pub(crate) operands: SmallVec<[Operand; 4]>,
    pub(crate) modifier: Option<InstructionModifier>,
}

impl Instruction {
    /// invalid instruction with its raw binary code
    fn invalid(code: u32) -> Instruction {
        Instruction {
            op: Opcode::INVALID(code),
            operands: SmallVec::<[Operand; 4]>::new(),
            modifier: None,
        }
    }

    pub fn get_opcode(&self) -> Opcode {
        self.op
    }
    pub fn get_operands(&self) -> &[Operand] {
        self.operands.as_slice()
    }
}

// Separate the code out of the TCB
impl Display for Instruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.op {
            Opcode::SOP1(opcode) => self.fmt_sop1_inst(f, opcode),
            Opcode::SOP2(opcode) => self.fmt_sop2_inst(f, opcode),
            Opcode::SOPK(opcode) => self.fmt_sopk_inst(f, opcode),
            Opcode::SOPP(opcode) => self.fmt_sopp_inst(f, opcode),
            Opcode::SOPC(opcode) => self.fmt_sopc_inst(f, opcode),
            Opcode::VOPC(opcode) => self.fmt_vopc_inst(f, opcode),
            Opcode::SMEM(opcode) => self.fmt_smem_inst(f, opcode),
            Opcode::VOP3AB(opcode) => self.fmt_vop3ab_inst(f, opcode),
            Opcode::VMEM(opcode) => self.fmt_vmem_inst(f, opcode),
            Opcode::MUBUF(opcode) => self.fmt_mubuf_inst(f, opcode),
            Opcode::VOP1(opcode) => self.fmt_vop1_inst(f, opcode),
            Opcode::VOP2(opcode) => self.fmt_vop2_inst(f, opcode),
            Opcode::VOP3P(opcode) => self.fmt_vop3p_inst(f, opcode),
            Opcode::LDSGDS(opcode) => self.fmt_ldsgds_inst(f, opcode),
            Opcode::INVALID(code) => write!(f, ".long {:#010x}", code),
            _ => {
                write!(f, "{}", self.op)?;
                for (idx, o) in self.get_operands().iter().enumerate() {
                    if idx == 0 {
                        write!(f, " {}", o)?;
                    } else {
                        write!(f, ", {}", o)?;
                    }
                }
                Ok(())
            }
        }
    }
}

impl Instruction {
    fn fmt_sop1_inst(&self, f: &mut Formatter<'_>, opcode: SOP1Opcode) -> std::fmt::Result {
        use SOP1InstExpectedOperands::*;
        use SOP1InstSideEffect::*;
        let op = self.get_operands();
        write!(f, "{}", self.op)?;
        let dst_size = opcode.dst_size();
        let src_size = opcode.src_size();
        let src_index = match SOP1InstSideEffect::from_opcode(opcode) {
            NoSideEffect => 1,
            ReadM0 | ReadScc | WriteScc => 2,
            WriteSccAndExec => 3,
        };
        match SOP1InstExpectedOperands::from_opcode(opcode) {
            NoDst => write!(f, " {:#.*}", src_size, &op[src_index]),
            NoSrc => write!(f, " {:#.*}", dst_size, &op[0]),
            AccessDstSrc => write!(
                f,
                " {:#.*}, {:#.*}",
                dst_size, &op[0], src_size, &op[src_index]
            ),
        }
    }

    fn fmt_sop2_inst(&self, f: &mut Formatter<'_>, opcode: SOP2Opcode) -> std::fmt::Result {
        use SOP2InstSideEffect::*;
        let op = self.get_operands();
        write!(f, "{}", self.op)?;
        let (dst_size, src0_size, src1_size) = opcode.reg_size();
        match SOP2InstSideEffect::from_opcode(opcode) {
            NoSideEffect => {
                write!(
                    f,
                    " {:.*}, {:#.*}, {:#.*}",
                    dst_size, &op[0], src0_size, &op[1], src1_size, &op[2]
                )
            }
            ReadScc | WriteScc | ReadWriteScc => {
                write!(
                    f,
                    " {:.*}, {:#.*}, {:#.*}",
                    dst_size, &op[0], src0_size, &op[2], src1_size, &op[3]
                )
            }
        }
    }

    fn fmt_sopk_inst(&self, f: &mut Formatter<'_>, opcode: SOPKOpcode) -> std::fmt::Result {
        use SOPKOpcode::*;
        let op = self.get_operands();
        write!(f, "{}", self.op)?;
        match opcode {
            S_VERSION => {
                write!(f, " {:#x}", &op[1])?;
            }
            S_CMOVK_I32 | S_MOVK_I32 | S_MULK_I32 => {
                if let Operand::Constant(v) = &op[1] {
                    write!(f, " {}, {:#x}", &op[0], (*v) as u16)?;
                } else {
                    Err(std::fmt::Error)?;
                }
            }
            S_WAITCNT_EXPCNT | S_WAITCNT_LGKMCNT | S_WAITCNT_VMCNT | S_WAITCNT_VSCNT => {
                write!(f, " {}, {:#x}", &op[1], &op[2])?;
            }
            S_CALL_B64 => {
                write!(f, " {:.2}, {}", &op[0], &op[1])?;
            }
            S_SUBVECTOR_LOOP_BEGIN | S_SUBVECTOR_LOOP_END => {
                write!(f, " {}, {:#}", &op[0], &op[1])?;
            }
            S_ADDK_I32 => {
                if let Operand::Constant(v) = &op[2] {
                    write!(f, " {}, {:#x}", &op[0], (*v) as u16)?;
                } else {
                    Err(std::fmt::Error)?;
                }
            }
            S_SETREG_B32 | S_SETREG_IMM32_B32 => {
                let hwreg = HWReg::from_operand(&op[1]).ok_or(std::fmt::Error)?;
                write!(f, " {}, {:#}", hwreg, &op[2])?;
            }
            S_GETREG_B32 => {
                let hwreg = HWReg::from_operand(&op[1]).ok_or(std::fmt::Error)?;
                write!(f, " {}, {}", &op[0], hwreg)?;
            }
            _ => {
                // S_CMPK_*
                if let Operand::Constant(v) = &op[2] {
                    write!(f, " {}, {:#x}", &op[1], (*v) as u16)?;
                } else {
                    Err(std::fmt::Error)?;
                }
            }
        }
        Ok(())
    }

    fn fmt_sopp_inst(&self, f: &mut Formatter<'_>, opcode: SOPPOpcode) -> std::fmt::Result {
        use SOPPOpcode::*;
        write!(f, "{}", self.op)?;
        match SOPPInstExpectedOperands::from_opcode(opcode) {
            SOPPInstExpectedOperands::NoOperand => {}
            SOPPInstExpectedOperands::ReadSimm => {
                let op = self.get_operands();
                match opcode {
                    S_WAITCNT => {
                        if let Operand::Constant(simm) = op[1] {
                            let fields = WaitcntFields::from_i32(simm);
                            write!(f, "{}", fields)?;
                        } else {
                            Err(std::fmt::Error)?;
                        }
                    }
                    S_SENDMSG | S_SENDMSGHALT => {
                        let send_msg = SendMsg::from_operand(&op[1]).ok_or(std::fmt::Error)?;
                        write!(f, " {}", send_msg)?;
                    }
                    S_INST_PREFETCH | S_CLAUSE => {
                        write!(f, " {:#x}", &op[1])?;
                    }
                    S_NOP | S_SETKILL | S_SETHALT | S_SLEEP | S_SETPRIO | S_TRAP
                    | S_INCPERFLEVEL | S_DECPERFLEVEL | S_WAITCNT_DEPCTR | S_ROUND_MODE
                    | S_DENORM_MODE | S_TTRACEDATA_IMM => {
                        write!(f, " {:#}", &op[1])?;
                    }
                    S_BRANCH
                    | S_CBRANCH_SCC0
                    | S_CBRANCH_SCC1
                    | S_CBRANCH_VCCZ
                    | S_CBRANCH_VCCNZ
                    | S_CBRANCH_EXECZ
                    | S_CBRANCH_EXECNZ
                    | S_CBRANCH_CDBGSYS
                    | S_CBRANCH_CDBGUSER
                    | S_CBRANCH_CDBGSYS_OR_USER
                    | S_CBRANCH_CDBGSYS_AND_USER => {
                        if let Operand::Constant(v) = &op[1] {
                            write!(f, " {}", (*v) as u16)?;
                        } else {
                            unreachable!();
                        }
                    }

                    _ => {
                        write!(f, " {}", &op[1])?;
                    }
                }
            }
            SOPPInstExpectedOperands::ConditionalBranch(_) => {
                let op = self.get_operands();
                match opcode {
                    S_CBRANCH_SCC0 | S_CBRANCH_SCC1 | S_CBRANCH_VCCZ | S_CBRANCH_VCCNZ
                    | S_CBRANCH_EXECZ | S_CBRANCH_EXECNZ => {
                        if let Operand::Constant(v) = &op[2] {
                            write!(f, " {}", (*v) as u16)?;
                        } else {
                            unreachable!();
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }
        Ok(())
    }

    fn fmt_sopc_inst(&self, f: &mut Formatter<'_>, opcode: SOPCOpcode) -> std::fmt::Result {
        let op = self.get_operands();
        let (inst_type, data_size) = opcode.get_type_size();
        let (s0_size, s1_size) = inst_type.src_size(data_size);
        write!(
            f,
            "{} {:#.*}, {:#.*}",
            opcode, s0_size, &op[1], s1_size, &op[2]
        )
    }

    fn fmt_vopc_inst(&self, f: &mut Formatter<'_>, opcode: VOPCOpcode) -> std::fmt::Result {
        if self.modifier.is_some() {
            write!(f, "{}_sdwa ", opcode)?;
        } else {
            write!(f, "{}_e32 ", opcode)?;
        }
        let op = self.get_operands();
        // for CMPX instructions which write exec, do not print the destination
        if !VOPCOpcodeFlags::from_opcode(opcode).write_exec() {
            write!(f, "{}, ", &op[0])?;
        }
        match &self.modifier {
            None => {
                let (inst_type, data_size) = opcode.get_type_size().ok_or(std::fmt::Error)?;
                let (s0_size, s1_size) = inst_type.src_size(data_size);
                write!(f, "{:#.*}, {:#.*}", s0_size, &op[1], s1_size, &op[2])?;
            }
            Some(InstructionModifier::Sdwa(m)) => {
                let m0 = m.parse_src0_modifier();
                let m1 = m.parse_src1_modifier();
                write!(f, "{}, {}", op[1].modified_by(m0), op[2].modified_by(m1))?;
                m.fmt_src0_sel(f)?;
                m.fmt_src1_sel(f)?;
            }
            _ => Err(std::fmt::Error)?,
        }
        Ok(())
    }

    fn fmt_smem_inst(&self, f: &mut Formatter<'_>, opcode: SMEMOpcode) -> std::fmt::Result {
        use SMEMInstExpectedOperands::*;
        write!(f, "{}", opcode)?;
        let op = self.get_operands();
        let fmt_offsets = |f: &mut Formatter<'_>, &soffset, &offset| match (soffset, offset) {
            (_, Operand::Constant(0)) => {
                write!(f, ", {}", soffset)
            }
            (Operand::SpecialScalarRegister(Operand::SPECIAL_REG_NULL), _) => {
                write!(f, ", {:#x}", offset)
            }
            (_, _) => {
                write!(f, "{}, {:#x}", soffset, offset)
            }
        };
        match opcode.expected_operands() {
            NoOperands => Ok(()),
            NoSrc => {
                write!(f, " {:.*}", opcode.data_size(), &op[0])
            }
            NoDst => {
                write!(f, " {}, {:.*}", &op[1], opcode.addr_size(), &op[2])?;
                fmt_offsets(f, &op[3], &op[4])?;
                Ok(())
            }
            AccessDstSrc => {
                write!(f, " {:.*},", opcode.data_size(), &op[0],)?;
                write!(f, " {:.*}", opcode.addr_size(), &op[1])?;
                fmt_offsets(f, &op[4], &op[5])?;
                match &op[3] {
                    Operand::Constant(0) => {}
                    Operand::Constant(1) => write!(f, " glc")?,
                    _ => Err(std::fmt::Error)?,
                }
                match &op[2] {
                    Operand::Constant(0) => {}
                    Operand::Constant(1) => write!(f, " dlc")?,
                    _ => Err(std::fmt::Error)?,
                }
                Ok(())
            }
        }
    }

    fn fmt_vop3ab_inst(&self, f: &mut Formatter<'_>, opcode: VOP3ABOpcode) -> std::fmt::Result {
        let op = self.get_operands();
        let (dst_size, src0_size, src1_size, src2_size) = opcode.reg_size();
        if opcode.explicit_64bit_encoding() {
            write!(f, "{}_e64", opcode)?;
        } else {
            write!(f, "{}", opcode)?;
        }
        let modifier = match &self.modifier {
            Some(InstructionModifier::Vop3(m)) => m,
            _ => Err(std::fmt::Error)?,
        };
        let src_count = opcode.src_count();
        let (dst, src);
        let reg_size: SmallVec<[_; 5]>;
        match VOP3ABDstType::from_opcode(opcode) {
            VOP3ABDstType::NoDst => {
                dst = &op[..0];
                src = &op[1..];
                reg_size = smallvec![src0_size, src1_size, src2_size];
            }
            VOP3ABDstType::VDstSDst => {
                dst = &op[..2];
                src = &op[2..];
                reg_size = smallvec![dst_size, 1, src0_size, src1_size, src2_size];
            }
            _ => {
                dst = &op[..1];
                src = &op[1..];
                reg_size = smallvec![dst_size, src0_size, src1_size, src2_size];
            }
        }
        let dst: SmallVec<[_; 2]> = dst.iter().map(|o| o.modified_by(None)).collect();
        let src: SmallVec<[_; 3]> = src
            .iter()
            .enumerate()
            .map(|(i, o)| o.modified_by(modifier.parse_src_modifier(i)))
            .collect();
        let operands = dst.iter().chain(src.iter());
        let operands: SmallVec<[_; 5]> = operands
            .zip(reg_size)
            .map(|(o, sz)| format!(" {:#.*}", sz as usize, o))
            .collect();
        write!(f, "{}", operands.join(","))?;
        if opcode.is_vop3b() {
            modifier.fmt_vop3b_dst_modifier(f)
        } else {
            modifier.fmt_vop3a_dst_modifier(f, src_count)
        }
    }

    fn fmt_vop3p_inst(&self, f: &mut Formatter<'_>, opcode: VOP3POpcode) -> std::fmt::Result {
        let modifier = match &self.modifier {
            Some(InstructionModifier::Vop3p(m)) => m,
            _ => Err(std::fmt::Error)?,
        };
        let op = self.get_operands();
        write!(f, "{} {}", opcode, op[0])?;
        op.iter().skip(1).try_for_each(|o| write!(f, ", {:#}", o))?;
        modifier.fmt_modifier(f, opcode)
    }

    fn fmt_vmem_inst(&self, f: &mut Formatter<'_>, opcode: VMEMOpcode) -> std::fmt::Result {
        use VMEMOpcode::*;
        use VMemInstructionType::*;
        let modifier = if let Some(InstructionModifier::VMem(m)) = &self.modifier {
            m
        } else {
            Err(std::fmt::Error)?
        };
        let op = self.get_operands();
        let inst_type = VMemInstructionType::from_opcode(opcode);
        let (vdst_size, vdata_size) = opcode.data_size();
        let vaddr_size = modifier.vaddr_size();
        let is_global = modifier.is_global().ok_or(std::fmt::Error)?;
        if is_global {
            write!(f, "{} ", opcode)?;
        } else {
            let flat_opcode = FlatOpcode::try_from(opcode as u32).map_err(|_| std::fmt::Error)?;
            write!(f, "{} ", flat_opcode)?;
        }
        let should_print_vdst = modifier.should_print_vdst(inst_type);
        let should_print_vaddr =
            !matches!(opcode, GLOBAL_LOAD_DWORD_ADDTID | GLOBAL_STORE_DWORD_ADDTID);
        let should_print_vdata = matches!(inst_type, Store | Atomic);
        let operands: SmallVec<[_; 3]> = op[0..3]
            .iter()
            .zip([vdst_size, vaddr_size, vdata_size])
            .zip([should_print_vdst, should_print_vaddr, should_print_vdata])
            .filter(|(_, enable)| *enable)
            .map(|((o, sz), _)| format!("{:.*}", sz, o))
            .collect();
        write!(f, "{}", operands.join(", "))?;
        modifier.fmt_saddr(f, &op[3])?;
        match self.operands[4] {
            Operand::Constant(0) => {}
            Operand::Constant(offset) => write!(f, " offset:{}", offset)?,
            _ => Err(std::fmt::Error)?,
        }
        write!(f, "{}", modifier)
    }

    fn fmt_mubuf_inst(&self, f: &mut Formatter<'_>, opcode: MUBUFOpcode) -> std::fmt::Result {
        use MUBUFInstructionType::*;
        write!(f, "{}", self.op)?;
        let reg_len = opcode.reg_len();
        let op = match MUBUFInstructionType::from_opcode(opcode) {
            BufferGL => {
                return Ok(());
            }
            BufferLoad | BufferAtomic => self.get_operands(),
            BufferStore => &self.get_operands()[1..],
        };
        let modifier = if let Some(InstructionModifier::Mubuf(m)) = self.modifier {
            m
        } else {
            Err(std::fmt::Error)?
        };
        if modifier.lds() == 0 {
            write!(f, " {:.*},", reg_len, &op[0])?;
        }
        match modifier.vaddr_size() {
            0 => write!(f, " off,")?,
            n => write!(f, " {:.*},", n as usize, &op[1])?,
        }
        write!(f, " {:.4}, {}", &op[2], &op[3])?;
        write!(f, "{}", modifier)
    }

    fn fmt_vop1_inst(&self, f: &mut Formatter<'_>, opcode: VOP1Opcode) -> std::fmt::Result {
        use VOP1Opcode::*;
        let op = self.get_operands();
        match &self.modifier {
            None => match opcode {
                V_CLREXCP | V_PIPEFLUSH | V_NOP => {
                    write!(f, "{}", self.op)
                }
                V_READFIRSTLANE_B32 => {
                    write!(f, "{} {}, {:#}", self.op, &op[0], &op[1])
                }
                _ => {
                    let (dst_size, src_size) = opcode.reg_size();
                    write!(
                        f,
                        "{}_e32 {:.*}, {:#.*}",
                        self.op, dst_size, &op[0], src_size, &op[1]
                    )
                }
            },
            Some(InstructionModifier::Sdwa(m)) => {
                let m0 = m.parse_src0_modifier();
                write!(f, "{}_sdwa {}, {}", opcode, &op[0], op[1].modified_by(m0))?;
                m.fmt_sdwa_dst_modifier(f)?;
                m.fmt_src0_sel(f)
            }
            Some(InstructionModifier::Dpp16(m)) => {
                let m0 = m.parse_src0_modifier();
                write!(f, "{}_dpp {}, {}", opcode, &op[0], op[1].modified_by(m0))?;
                write!(f, " {}", m)
            }
            _ => Err(std::fmt::Error)?,
        }
    }

    fn fmt_vop2_inst(&self, f: &mut Formatter<'_>, opcode: VOP2Opcode) -> std::fmt::Result {
        use VOP2Opcode::*;
        if let Some(m) = &self.modifier {
            return self.fmt_vop2_inst_with_modifier(f, opcode, m);
        }
        match opcode {
            V_FMAAK_F16 | V_FMAAK_F32 | V_FMAMK_F16 | V_FMAMK_F32 => {
                write!(f, "{} ", self.op)?;
            }
            _ => {
                write!(f, "{}_e32 ", self.op)?;
            }
        }
        write!(f, "{:#}", self.operands[0])?;
        self.operands
            .iter()
            .skip(1)
            .try_for_each(|o| write!(f, ", {:#}", o))
    }

    fn fmt_vop2_inst_with_modifier(
        &self,
        f: &mut Formatter<'_>,
        opcode: VOP2Opcode,
        modifier: &InstructionModifier,
    ) -> std::fmt::Result {
        use InstructionModifier::*;
        use VOP2InstSideEffect::*;
        let (m0, m1) = match modifier {
            Sdwa(sdwa) => {
                write!(f, "{}_sdwa", self.op)?;
                (sdwa.parse_src0_modifier(), sdwa.parse_src1_modifier())
            }
            Dpp16(dpp) => {
                write!(f, "{}_dpp", self.op)?;
                (dpp.parse_src0_modifier(), dpp.parse_src1_modifier())
            }
            _ => Err(std::fmt::Error)?,
        };
        let src_modifiers: SmallVec<[_; 4]> = match VOP2InstSideEffect::from_opcode(opcode) {
            NoSideEffect => smallvec![Some(m0), Some(m1)],
            ReadVcc => smallvec![None, Some(m0), Some(m1)],
            ReadAndWriteVcc => smallvec![None, Some(m0), Some(m1), None],
        };
        write!(f, " {:#}", self.operands[0])?;
        self.operands
            .iter()
            .skip(1)
            .zip(src_modifiers)
            .try_for_each(|(o, m)| match m {
                Some(m) => {
                    write!(f, ", {}", o.modified_by(m))
                }
                None => write!(f, ", {:#}", o),
            })?;
        match modifier {
            Sdwa(sdwa) => {
                sdwa.fmt_sdwa_dst_modifier(f)?;
                sdwa.fmt_src0_sel(f)?;
                sdwa.fmt_src1_sel(f)?;
            }
            Dpp16(dpp) => {
                write!(f, " {}", dpp)?;
            }
            _ => Err(std::fmt::Error)?,
        }
        Ok(())
    }

    fn fmt_ldsgds_inst(&self, f: &mut Formatter<'_>, opcode: LDSGDSOpcode) -> std::fmt::Result {
        use DSInstSideEffect::*;
        write!(f, "{}", opcode)?;
        let op = self.get_operands();
        let reg_size = opcode.reg_size();
        let (dst_count, addr_count, data_count) = opcode.expected_operands();
        let side_effect = opcode.side_effect();
        let dst_offset = 0usize;
        let addr_offset = match side_effect {
            NoSideEffect => 1usize,
            ReadM0 => 2usize,
        };
        let data_offset = addr_offset + addr_count as usize;
        let operands = [
            (dst_count > 0).then_some(dst_offset),
            (addr_count > 0).then_some(addr_offset),
            (data_count > 0).then_some(data_offset),
            (data_count > 1).then_some(data_offset + 1),
        ];
        let op_size = [dst_count as usize * reg_size, 1, reg_size, reg_size];
        let operands: SmallVec<[_; 4]> = op_size
            .iter()
            .zip(operands)
            .filter_map(|(sz, x)| x.map(|o| (sz, o)))
            .collect();
        if !operands.is_empty() {
            let (sz, i) = operands[0];
            write!(f, " {:.*}", sz, op[i])?;
            operands
                .iter()
                .skip(1)
                .try_for_each(|&(sz, i)| write!(f, ", {:.*}", sz, op[i]))?;
        }
        if let Some(InstructionModifier::Ds(m)) = self.modifier {
            m.fmt(f, opcode.offset_count())
        } else {
            Err(std::fmt::Error)
        }
    }
}

pub(crate) enum DSInstSideEffect {
    NoSideEffect,
    ReadM0,
}

impl LDSGDSOpcode {
    pub(crate) fn side_effect(&self) -> DSInstSideEffect {
        use LDSGDSOpcode::*;
        match self {
            DS_GWS_BARRIER
            | DS_GWS_INIT
            | DS_GWS_SEMA_BR
            | DS_GWS_SEMA_P
            | DS_GWS_SEMA_RELEASE_ALL
            | DS_GWS_SEMA_V
            | DS_WRITE_ADDTID_B32
            | DS_APPEND
            | DS_CONSUME
            | DS_READ_ADDTID_B32 => DSInstSideEffect::ReadM0,
            _ => DSInstSideEffect::NoSideEffect,
        }
    }

    /// Returns expected count of dst, addr, data
    pub(crate) fn expected_operands(&self) -> (u8, u8, u8) {
        use LDSGDSOpcode::*;
        match self {
            // read
            DS_READ_I8 | DS_READ_I8_D16 | DS_READ_I8_D16_HI | DS_READ_U8 | DS_READ_U8_D16
            | DS_READ_U8_D16_HI | DS_READ_I16 | DS_READ_U16 | DS_READ_U16_D16
            | DS_READ_U16_D16_HI | DS_READ_B32 | DS_READ_B64 | DS_READ_B96 | DS_READ_B128
            | DS_ORDERED_COUNT | DS_SWIZZLE_B32 => (1, 1, 0),
            // read (two pieces of data)
            DS_READ2_B32 | DS_READ2ST64_B32 | DS_READ2_B64 | DS_READ2ST64_B64 => (2, 1, 0),
            // write
            DS_WRITE_B8 | DS_WRITE_B8_D16_HI | DS_WRITE_B16 | DS_WRITE_B16_D16_HI
            | DS_WRITE_B32 | DS_WRITE_B64 | DS_WRITE_B96 | DS_WRITE_B128 => (0, 1, 1),
            // write (two pieces of data)
            DS_WRITE2_B32 | DS_WRITE2ST64_B32 | DS_WRITE2_B64 | DS_WRITE2ST64_B64 => (0, 1, 2),
            // write-exchange
            DS_WRXCHG_RTN_B32 | DS_WRXCHG_RTN_B64 => (1, 1, 1),
            // write-exchange (two pieces of data)
            DS_WRXCHG2_RTN_B32
            | DS_WRXCHG2_RTN_B64
            | DS_WRXCHG2ST64_RTN_B32
            | DS_WRXCHG2ST64_RTN_B64 => (2, 1, 2),
            // atomic operation with two data, discard return value
            DS_CMPST_B32 | DS_CMPST_B64 | DS_CMPST_F32 | DS_CMPST_F64 | DS_MSKOR_B32
            | DS_MSKOR_B64 => (0, 1, 2),
            // atomic operation with two data, return value to `vdst`
            DS_CMPST_RTN_B32 | DS_CMPST_RTN_B64 | DS_CMPST_RTN_F32 | DS_CMPST_RTN_F64
            | DS_MSKOR_RTN_B32 | DS_MSKOR_RTN_B64 | DS_WRAP_RTN_B32 => (1, 1, 2),
            // gws operation with data
            DS_GWS_BARRIER | DS_GWS_INIT | DS_GWS_SEMA_BR => (0, 1, 0),
            // gws operation without data
            DS_GWS_SEMA_P | DS_GWS_SEMA_RELEASE_ALL | DS_GWS_SEMA_V => (0, 0, 0),
            DS_NOP => (0, 0, 0),
            DS_WRITE_ADDTID_B32 => (0, 0, 1),
            DS_APPEND | DS_CONSUME | DS_READ_ADDTID_B32 => (1, 0, 0),
            // atomic operation, discard return value
            DS_ADD_F32 | DS_ADD_U32 | DS_ADD_U64 | DS_AND_B32 | DS_AND_B64 | DS_DEC_U32
            | DS_DEC_U64 | DS_INC_U32 | DS_INC_U64 | DS_MAX_U32 | DS_MAX_U64 | DS_MAX_I32
            | DS_MAX_I64 | DS_MAX_F32 | DS_MAX_F64 | DS_MIN_U32 | DS_MIN_U64 | DS_MIN_I32
            | DS_MIN_I64 | DS_MIN_F32 | DS_MIN_F64 | DS_OR_B32 | DS_OR_B64 | DS_XOR_B32
            | DS_XOR_B64 | DS_RSUB_U32 | DS_RSUB_U64 | DS_SUB_U32 | DS_SUB_U64 => (0, 1, 1),
            // atomic operation, return value to `vdst`
            _ => (1, 1, 1),
        }
    }

    fn offset_count(&self) -> u8 {
        use LDSGDSOpcode::*;
        match self {
            DS_NOP => 0,
            DS_READ2_B32
            | DS_READ2ST64_B32
            | DS_READ2_B64
            | DS_READ2ST64_B64
            | DS_WRITE2_B32
            | DS_WRITE2ST64_B32
            | DS_WRITE2_B64
            | DS_WRITE2ST64_B64
            | DS_WRXCHG2_RTN_B32
            | DS_WRXCHG2_RTN_B64
            | DS_WRXCHG2ST64_RTN_B32
            | DS_WRXCHG2ST64_RTN_B64 => 2,
            _ => 1,
        }
    }

    fn reg_size(&self) -> usize {
        use LDSGDSOpcode::*;
        match self {
            DS_ADD_U64
            | DS_ADD_RTN_U64
            | DS_AND_B64
            | DS_AND_RTN_B64
            | DS_CMPST_B64
            | DS_CMPST_RTN_B64
            | DS_CMPST_F64
            | DS_CMPST_RTN_F64
            | DS_CONDXCHG32_RTN_B64
            | DS_DEC_U64
            | DS_DEC_RTN_U64
            | DS_INC_U64
            | DS_INC_RTN_U64
            | DS_MAX_F64
            | DS_MAX_RTN_F64
            | DS_MAX_I64
            | DS_MAX_RTN_I64
            | DS_MIN_F64
            | DS_MIN_RTN_F64
            | DS_MIN_I64
            | DS_MIN_RTN_I64
            | DS_MSKOR_B64
            | DS_MSKOR_RTN_B64
            | DS_OR_B64
            | DS_OR_RTN_B64
            | DS_XOR_B64
            | DS_XOR_RTN_B64
            | DS_RSUB_U64
            | DS_RSUB_RTN_U64
            | DS_SUB_U64
            | DS_SUB_RTN_U64
            | DS_WRXCHG2_RTN_B64
            | DS_WRXCHG2ST64_RTN_B64
            | DS_WRXCHG_RTN_B64
            | DS_READ_B64
            | DS_WRITE_B64
            | DS_READ2_B64
            | DS_READ2ST64_B64
            | DS_WRITE2_B64
            | DS_WRITE2ST64_B64 => 2,
            DS_READ_B96 | DS_WRITE_B96 => 3,
            DS_READ_B128 | DS_WRITE_B128 => 4,
            _ => 1,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum VMemInstructionType {
    Load,
    Store,
    Atomic,
}

impl VMemInstructionType {
    pub(crate) fn from_opcode(opcode: VMEMOpcode) -> Self {
        use VMEMOpcode::*;
        use VMemInstructionType::*;
        match opcode {
            GLOBAL_STORE_DWORD_ADDTID
            | GLOBAL_STORE_BYTE
            | GLOBAL_STORE_BYTE_D16_HI
            | GLOBAL_STORE_SHORT
            | GLOBAL_STORE_SHORT_D16_HI
            | GLOBAL_STORE_DWORD
            | GLOBAL_STORE_DWORDX2
            | GLOBAL_STORE_DWORDX4
            | GLOBAL_STORE_DWORDX3 => Store,
            GLOBAL_LOAD_DWORD_ADDTID
            | GLOBAL_LOAD_UBYTE
            | GLOBAL_LOAD_SBYTE
            | GLOBAL_LOAD_USHORT
            | GLOBAL_LOAD_SSHORT
            | GLOBAL_LOAD_DWORD
            | GLOBAL_LOAD_DWORDX2
            | GLOBAL_LOAD_DWORDX4
            | GLOBAL_LOAD_DWORDX3
            | GLOBAL_LOAD_UBYTE_D16
            | GLOBAL_LOAD_UBYTE_D16_HI
            | GLOBAL_LOAD_SBYTE_D16
            | GLOBAL_LOAD_SBYTE_D16_HI
            | GLOBAL_LOAD_SHORT_D16
            | GLOBAL_LOAD_SHORT_D16_HI => Load,
            _ => Atomic,
        }
    }
}

impl VMEMOpcode {
    // size for (vdst, vdata)
    pub(crate) fn data_size(&self) -> (usize, usize) {
        use VMEMOpcode::*;
        match self {
            GLOBAL_LOAD_DWORDX2 => (2, 0),
            GLOBAL_LOAD_DWORDX3 => (3, 0),
            GLOBAL_LOAD_DWORDX4 => (4, 0),
            GLOBAL_STORE_DWORDX2 => (0, 2),
            GLOBAL_STORE_DWORDX3 => (0, 3),
            GLOBAL_STORE_DWORDX4 => (0, 4),
            GLOBAL_ATOMIC_ADD_X2
            | GLOBAL_ATOMIC_AND_X2
            | GLOBAL_ATOMIC_DEC_X2
            | GLOBAL_ATOMIC_FMAX_X2
            | GLOBAL_ATOMIC_FMIN_X2
            | GLOBAL_ATOMIC_INC_X2
            | GLOBAL_ATOMIC_OR_X2
            | GLOBAL_ATOMIC_SMAX_X2
            | GLOBAL_ATOMIC_SMIN_X2
            | GLOBAL_ATOMIC_SUB_X2
            | GLOBAL_ATOMIC_SWAP_X2
            | GLOBAL_ATOMIC_UMAX_X2
            | GLOBAL_ATOMIC_UMIN_X2
            | GLOBAL_ATOMIC_XOR_X2 => (2, 2),
            GLOBAL_ATOMIC_CMPSWAP | GLOBAL_ATOMIC_FCMPSWAP => (1, 2),
            GLOBAL_ATOMIC_CMPSWAP_X2 | GLOBAL_ATOMIC_FCMPSWAP_X2 => (2, 4),
            _ => (1, 1),
        }
    }
}

pub(crate) enum MUBUFInstructionType {
    BufferAtomic,
    BufferLoad,
    BufferStore,
    BufferGL,
}

impl MUBUFInstructionType {
    pub(crate) fn from_opcode(opcode: MUBUFOpcode) -> Self {
        use MUBUFInstructionType::*;
        use MUBUFOpcode::*;
        match opcode {
            BUFFER_LOAD_FORMAT_X
            | BUFFER_LOAD_FORMAT_XY
            | BUFFER_LOAD_FORMAT_XYZ
            | BUFFER_LOAD_FORMAT_XYZW
            | BUFFER_LOAD_UBYTE
            | BUFFER_LOAD_SBYTE
            | BUFFER_LOAD_USHORT
            | BUFFER_LOAD_SSHORT
            | BUFFER_LOAD_DWORD
            | BUFFER_LOAD_DWORDX2
            | BUFFER_LOAD_DWORDX4
            | BUFFER_LOAD_DWORDX3
            | BUFFER_LOAD_UBYTE_D16
            | BUFFER_LOAD_UBYTE_D16_HI
            | BUFFER_LOAD_SBYTE_D16
            | BUFFER_LOAD_SBYTE_D16_HI
            | BUFFER_LOAD_SHORT_D16
            | BUFFER_LOAD_SHORT_D16_HI
            | BUFFER_LOAD_FORMAT_D16_HI_X
            | BUFFER_LOAD_FORMAT_D16_X
            | BUFFER_LOAD_FORMAT_D16_XY
            | BUFFER_LOAD_FORMAT_D16_XYZ
            | BUFFER_LOAD_FORMAT_D16_XYZW => BufferLoad,
            BUFFER_STORE_FORMAT_X
            | BUFFER_STORE_FORMAT_XY
            | BUFFER_STORE_FORMAT_XYZ
            | BUFFER_STORE_FORMAT_XYZW
            | BUFFER_STORE_BYTE
            | BUFFER_STORE_BYTE_D16_HI
            | BUFFER_STORE_SHORT
            | BUFFER_STORE_SHORT_D16_HI
            | BUFFER_STORE_DWORD
            | BUFFER_STORE_DWORDX2
            | BUFFER_STORE_DWORDX4
            | BUFFER_STORE_DWORDX3
            | BUFFER_STORE_FORMAT_D16_HI_X
            | BUFFER_STORE_FORMAT_D16_X
            | BUFFER_STORE_FORMAT_D16_XY
            | BUFFER_STORE_FORMAT_D16_XYZ
            | BUFFER_STORE_FORMAT_D16_XYZW => BufferStore,
            BUFFER_GL0_INV | BUFFER_GL1_INV => BufferGL,
            _ => BufferAtomic,
        }
    }
}

impl MUBUFOpcode {
    fn reg_len(&self) -> usize {
        use MUBUFOpcode::*;
        match self {
            BUFFER_ATOMIC_ADD_X2
            | BUFFER_ATOMIC_SUB_X2
            | BUFFER_ATOMIC_AND_X2
            | BUFFER_ATOMIC_OR_X2
            | BUFFER_ATOMIC_XOR_X2
            | BUFFER_ATOMIC_CMPSWAP
            | BUFFER_ATOMIC_FCMPSWAP
            | BUFFER_ATOMIC_SWAP_X2
            | BUFFER_ATOMIC_FMAX_X2
            | BUFFER_ATOMIC_FMIN_X2
            | BUFFER_ATOMIC_INC_X2
            | BUFFER_ATOMIC_DEC_X2
            | BUFFER_ATOMIC_SMAX_X2
            | BUFFER_ATOMIC_SMIN_X2
            | BUFFER_ATOMIC_UMAX_X2
            | BUFFER_ATOMIC_UMIN_X2
            | BUFFER_LOAD_DWORDX2
            | BUFFER_STORE_DWORDX2 => 2,
            BUFFER_LOAD_DWORDX3 | BUFFER_STORE_DWORDX3 => 3,
            BUFFER_ATOMIC_CMPSWAP_X2
            | BUFFER_ATOMIC_FCMPSWAP_X2
            | BUFFER_LOAD_DWORDX4
            | BUFFER_STORE_DWORDX4 => 4,
            _ => 1,
        }
    }
}

/// `SOP1` instructions may have side effects on special registers:
/// - read `SCC` as the scalar condition
/// - write `SCC` as `DST != 0`
/// - update `EXEC`, which is the execute mask
/// - read `M0` as an index for SGPR
pub(crate) enum SOP1InstSideEffect {
    NoSideEffect,
    ReadScc,
    WriteScc,
    WriteSccAndExec,
    ReadM0,
}

impl SOP1InstSideEffect {
    pub(crate) fn from_opcode(opcode: SOP1Opcode) -> Self {
        use SOP1InstSideEffect::*;
        use SOP1Opcode::*;
        match opcode {
            S_CMOV_B32 | S_CMOV_B64 => ReadScc,
            S_NOT_B32 | S_WQM_B32 | S_QUADMASK_B32 | S_ABS_I32 | S_BCNT0_I32_B32
            | S_BCNT1_I32_B32 | S_NOT_B64 | S_WQM_B64 | S_QUADMASK_B64 | S_BCNT0_I32_B64
            | S_BCNT1_I32_B64 => WriteScc,
            S_MOVRELD_B32 | S_MOVRELS_B32 | S_MOVRELSD_2_B32 | S_MOVRELD_B64 | S_MOVRELS_B64 => {
                ReadM0
            }
            S_AND_SAVEEXEC_B64 | S_OR_SAVEEXEC_B64 | S_XOR_SAVEEXEC_B64 | S_ANDN2_SAVEEXEC_B64
            | S_ORN2_SAVEEXEC_B64 | S_NAND_SAVEEXEC_B64 | S_NOR_SAVEEXEC_B64
            | S_XNOR_SAVEEXEC_B64 | S_ANDN1_SAVEEXEC_B64 | S_ORN1_SAVEEXEC_B64
            | S_ANDN1_WREXEC_B64 | S_ANDN2_WREXEC_B64 | S_AND_SAVEEXEC_B32 | S_OR_SAVEEXEC_B32
            | S_XOR_SAVEEXEC_B32 | S_ANDN2_SAVEEXEC_B32 | S_ORN2_SAVEEXEC_B32
            | S_NAND_SAVEEXEC_B32 | S_NOR_SAVEEXEC_B32 | S_XNOR_SAVEEXEC_B32
            | S_ANDN1_SAVEEXEC_B32 | S_ORN1_SAVEEXEC_B32 | S_ANDN1_WREXEC_B32
            | S_ANDN2_WREXEC_B32 => WriteSccAndExec,
            _ => NoSideEffect,
        }
    }
}

pub(crate) enum SOP1InstExpectedOperands {
    AccessDstSrc,
    NoDst,
    NoSrc,
}

impl SOP1InstExpectedOperands {
    pub(crate) fn from_opcode(opcode: SOP1Opcode) -> Self {
        use SOP1InstExpectedOperands::*;
        use SOP1Opcode::*;
        match opcode {
            S_GETPC_B64 => NoSrc,
            S_SETPC_B64 | S_RFE_B64 => NoDst,
            _ => AccessDstSrc,
        }
    }
}

impl SOP1Opcode {
    fn dst_size(&self) -> usize {
        use SOP1Opcode::*;
        match self {
            S_SETPC_B64 | S_RFE_B64 => 0,
            S_GETPC_B64 | S_CMOV_B64 | S_NOT_B64 | S_WQM_B64 | S_QUADMASK_B64 | S_BCNT0_I32_B64
            | S_BCNT1_I32_B64 | S_MOVRELD_B64 | S_MOVRELS_B64 | S_AND_SAVEEXEC_B64
            | S_OR_SAVEEXEC_B64 | S_XOR_SAVEEXEC_B64 | S_ANDN2_SAVEEXEC_B64
            | S_ORN2_SAVEEXEC_B64 | S_NAND_SAVEEXEC_B64 | S_NOR_SAVEEXEC_B64
            | S_XNOR_SAVEEXEC_B64 | S_ANDN1_SAVEEXEC_B64 | S_ORN1_SAVEEXEC_B64
            | S_ANDN1_WREXEC_B64 | S_ANDN2_WREXEC_B64 | S_MOV_B64 | S_BREV_B64 | S_BITSET0_B64
            | S_BITSET1_B64 | S_SWAPPC_B64 => 2,
            _ => 1,
        }
    }
    fn src_size(&self) -> usize {
        use SOP1Opcode::*;
        match self {
            S_GETPC_B64 => 0,
            S_SETPC_B64 | S_RFE_B64 | S_CMOV_B64 | S_NOT_B64 | S_WQM_B64 | S_QUADMASK_B64
            | S_BCNT0_I32_B64 | S_BCNT1_I32_B64 | S_MOVRELD_B64 | S_MOVRELS_B64
            | S_AND_SAVEEXEC_B64 | S_OR_SAVEEXEC_B64 | S_XOR_SAVEEXEC_B64
            | S_ANDN2_SAVEEXEC_B64 | S_ORN2_SAVEEXEC_B64 | S_NAND_SAVEEXEC_B64
            | S_NOR_SAVEEXEC_B64 | S_XNOR_SAVEEXEC_B64 | S_ANDN1_SAVEEXEC_B64
            | S_ORN1_SAVEEXEC_B64 | S_ANDN1_WREXEC_B64 | S_ANDN2_WREXEC_B64 | S_FF0_I32_B64
            | S_FF1_I32_B64 | S_FLBIT_I32_B64 | S_FLBIT_I32_I64 | S_MOV_B64 | S_BREV_B64
            | S_BITSET0_B64 | S_BITSET1_B64 | S_SWAPPC_B64 => 2,
            _ => 1,
        }
    }
}

/// `SOP2` instructions may read or write `SCC` bit
pub(crate) enum SOP2InstSideEffect {
    NoSideEffect,
    WriteScc,
    ReadWriteScc,
    ReadScc,
}

impl SOP2InstSideEffect {
    pub(crate) fn from_opcode(opcode: SOP2Opcode) -> Self {
        use SOP2InstSideEffect::*;
        use SOP2Opcode::*;
        match opcode {
            // read SCC as condition
            S_CSELECT_B32 | S_CSELECT_B64 => ReadScc,
            // read & write SCC as carrier
            S_ADDC_U32 | S_SUBB_U32 => ReadWriteScc,
            // write SCC as carrier
            | S_ADD_U32 | S_SUB_U32 | S_ADD_I32 | S_SUB_I32
            // write SCC as comparison result
            | S_MIN_I32 | S_MIN_U32 | S_MAX_I32 | S_MAX_U32
            // write SCC as result not zero
            | S_AND_B32 | S_AND_B64 | S_OR_B32 | S_OR_B64 | S_XOR_B32 | S_XOR_B64
            | S_ANDN2_B32 | S_ANDN2_B64 | S_ORN2_B32 | S_ORN2_B64 | S_NAND_B32 | S_NAND_B64
            | S_NOR_B32 | S_NOR_B64 | S_XNOR_B32 | S_XNOR_B64
            | S_LSHL_B32 | S_LSHL_B64 | S_LSHR_B32 | S_LSHR_B64 | S_ASHR_I32 | S_ASHR_I64
            | S_BFE_U32 | S_BFE_I32 | S_BFE_U64 | S_BFE_I64 | S_ABSDIFF_I32
            // write SCC as overflow
            | S_LSHL1_ADD_U32 | S_LSHL2_ADD_U32 | S_LSHL3_ADD_U32 | S_LSHL4_ADD_U32 => WriteScc,
            _ => NoSideEffect,
        }
    }
}

impl SOP2Opcode {
    pub(crate) const fn reg_size(&self) -> (usize, usize, usize) {
        use SOP2Opcode::*;
        match self {
            S_CSELECT_B64 | S_BFM_B64 | S_AND_B64 | S_OR_B64 | S_XOR_B64 | S_ANDN2_B64
            | S_ORN2_B64 | S_NAND_B64 | S_NOR_B64 | S_XNOR_B64 => (2, 2, 2),
            S_BFE_U64 | S_BFE_I64 | S_LSHL_B64 | S_LSHR_B64 | S_ASHR_I64 => (2, 2, 1),
            _ => (1, 1, 1),
        }
    }
}

impl VOP1Opcode {
    pub(crate) const fn reg_size(&self) -> (usize, usize) {
        use VOP1Opcode::*;
        match self {
            V_CLREXCP | V_PIPEFLUSH | V_NOP => (1, 0),
            V_READFIRSTLANE_B32 => (1, 1),
            V_MOVRELD_B32 | V_MOVRELS_B32 | V_MOVRELSD_B32 | V_MOVRELSD_2_B32 | V_SWAPREL_B32 => {
                (1, 1)
            }
            V_CEIL_F64 | V_FLOOR_F64 | V_FRACT_F64 | V_FREXP_MANT_F64 | V_RCP_F64 | V_RNDNE_F64
            | V_RSQ_F64 | V_SQRT_F64 | V_TRUNC_F64 => (2, 2),
            V_CVT_F32_F64 | V_CVT_I32_F64 | V_CVT_U32_F64 | V_FREXP_EXP_I32_F64 => (1, 2),
            V_CVT_F64_F32 | V_CVT_F64_I32 | V_CVT_F64_U32 => (2, 1),
            _ => (1, 1),
        }
    }
}

/// `VOP2` instructions may have side effects to read or write VCC as carrier or condition
pub(crate) enum VOP2InstSideEffect {
    NoSideEffect,
    ReadVcc,
    ReadAndWriteVcc,
}

impl VOP2InstSideEffect {
    pub(crate) fn from_opcode(opcode: VOP2Opcode) -> Self {
        use VOP2InstSideEffect::*;
        use VOP2Opcode::*;
        match opcode {
            V_ADD_CO_CI_U32 | V_SUB_CO_CI_U32 | V_SUBREV_CO_CI_U32 => ReadAndWriteVcc,
            V_CNDMASK_B32 => ReadVcc,
            _ => NoSideEffect,
        }
    }
}

/// Some `VOP2` instructions may contains an extra literal constant
/// and have 3 source operands in total.
/// The extra operand may be placed in the second or third position.
pub(crate) enum VOP2InstExpectedOperands {
    TwoSrcs,
    ExtraSimmAsSrc1,
    ExtraSimmAsSrc2,
}

impl VOP2InstExpectedOperands {
    pub(crate) fn from_opcode(opcode: VOP2Opcode) -> Self {
        use VOP2InstExpectedOperands::*;
        use VOP2Opcode::*;
        match opcode {
            V_FMAMK_F16 | V_FMAMK_F32 => ExtraSimmAsSrc1,
            V_FMAAK_F16 | V_FMAAK_F32 => ExtraSimmAsSrc2,
            _ => TwoSrcs,
        }
    }
}

enum VMemSegment {
    Flat,
    Scratch,
    Global,
}

impl TryFrom<u32> for VMemSegment {
    type Error = ();
    fn try_from(value: u32) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Flat),
            1 => Ok(Self::Scratch),
            2 => Ok(Self::Global),
            _ => Err(()),
        }
    }
}

bitfield! {
    #[derive(Clone, Debug)]
    pub struct VMEMModifier(u32);
    pub dlc, _: 12, 12;
    pub lds, _: 13, 13;
    pub seg, _: 15, 14;
    pub glc, _: 16, 16;
    pub slc, _: 17, 17;
    pub saddr_null, set_saddr_null: 18, 18;
}

impl VMEMModifier {
    const LO_MASK: u32 = 0x3ffff;
    const SADDR_MASK: u32 = 0xff0000;
    const SADDR_OFFSET: u32 = 16;

    fn from_instruction(lo: u32, hi: u32) -> Self {
        let saddr = ((hi & Self::SADDR_MASK) >> Self::SADDR_OFFSET) as u8;
        let mut modifier = Self(lo & Self::LO_MASK);
        if matches!(
            saddr,
            Operand::SPECIAL_REG_NULL | Operand::SPECIAL_REG_EXEC_HI
        ) {
            modifier.set_saddr_null(1);
        }
        modifier
    }

    fn is_global(&self) -> Option<bool> {
        let seg = VMemSegment::try_from(self.seg()).ok()?;
        Some(matches!(seg, VMemSegment::Global))
    }

    fn vaddr_size(&self) -> usize {
        if self.saddr_null() == 0 {
            1
        } else {
            2
        }
    }

    fn should_print_vdst(&self, inst_type: VMemInstructionType) -> bool {
        use VMemInstructionType::*;
        match inst_type {
            Load => self.lds() == 0,
            Atomic => self.glc() != 0,
            Store => false,
        }
    }

    fn fmt_saddr(&self, f: &mut Formatter<'_>, saddr: &Operand) -> std::fmt::Result {
        if self.is_global().ok_or(std::fmt::Error)? {
            if self.saddr_null() != 0 {
                write!(f, ", off")?;
            } else {
                write!(f, ", {:.2}", saddr)?;
            }
        }
        Ok(())
    }
}

impl Display for VMEMModifier {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let flags = [
            (self.glc(), "glc"),
            (self.slc(), "slc"),
            (self.dlc(), "dlc"),
            (self.lds(), "lds"),
        ];
        flags
            .iter()
            .filter(|(flag, _)| *flag != 0)
            .try_for_each(|(_, name)| write!(f, " {}", name))
    }
}

pub(crate) enum SMEMInstExpectedOperands {
    NoOperands,
    NoSrc,
    NoDst,
    AccessDstSrc,
}

impl SMEMOpcode {
    pub(crate) fn data_size(&self) -> usize {
        use SMEMOpcode::*;
        match self {
            S_LOAD_DWORD | S_BUFFER_LOAD_DWORD => 1,
            S_LOAD_DWORDX2 | S_BUFFER_LOAD_DWORDX2 => 2,
            S_LOAD_DWORDX4 | S_BUFFER_LOAD_DWORDX4 => 4,
            S_LOAD_DWORDX8 | S_BUFFER_LOAD_DWORDX8 => 8,
            S_LOAD_DWORDX16 | S_BUFFER_LOAD_DWORDX16 => 16,
            S_MEMTIME | S_MEMREALTIME => 2,
            S_GL1_INV | S_DCACHE_INV | S_ATC_PROBE | S_ATC_PROBE_BUFFER => 0,
        }
    }

    fn addr_size(&self) -> usize {
        use SMEMOpcode::*;
        match self {
            S_LOAD_DWORD | S_LOAD_DWORDX2 | S_LOAD_DWORDX4 | S_LOAD_DWORDX8 | S_LOAD_DWORDX16
            | S_ATC_PROBE => 2,
            S_BUFFER_LOAD_DWORD
            | S_BUFFER_LOAD_DWORDX2
            | S_BUFFER_LOAD_DWORDX4
            | S_BUFFER_LOAD_DWORDX8
            | S_BUFFER_LOAD_DWORDX16
            | S_ATC_PROBE_BUFFER => 4,
            _ => 0,
        }
    }

    pub(crate) fn expected_operands(&self) -> SMEMInstExpectedOperands {
        use SMEMInstExpectedOperands::*;
        use SMEMOpcode::*;
        match self {
            S_GL1_INV | S_DCACHE_INV => NoOperands,
            S_MEMTIME | S_MEMREALTIME => NoSrc,
            S_ATC_PROBE | S_ATC_PROBE_BUFFER => NoDst, // `sdata` is imm instead of sgpr
            _ => AccessDstSrc,
        }
    }
}

bitfield! {
    #[derive(Clone, Copy, Debug)]
    pub struct MUBUFModifier(u32);
    offset, _: 11, 0;
    offen, _: 12, 12;
    idxen, _: 13, 13;
    glc, _: 14, 14;
    dlc, _: 15, 15;
    lds, _: 16, 16;
    slc, _: 22, 22;
}

impl MUBUFModifier {
    const LO_MASK: u32 = 0x1ffff; // 17 bits
    const HI_MASK: u32 = 0x400000; // bit[22] = 1

    fn from_instrcution(lo: u32, hi: u32) -> Self {
        Self((lo & Self::LO_MASK) | (hi & Self::HI_MASK))
    }

    fn vaddr_size(&self) -> u8 {
        (self.offen() + self.idxen()) as u8
    }
}

impl Display for MUBUFModifier {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.idxen() != 0 {
            write!(f, " idxen")?;
        }
        if self.offen() != 0 {
            write!(f, " offen")?;
        }
        match self.offset() {
            0 => {}
            x => write!(f, " offset:{}", x)?,
        }
        let flags = [
            (self.glc(), "glc"),
            (self.slc(), "slc"),
            (self.dlc(), "dlc"),
            (self.lds(), "lds"),
        ];
        flags
            .iter()
            .filter(|(flag, _)| *flag != 0)
            .try_for_each(|(_, name)| write!(f, " {}", name))
    }
}

bitfield! {
    struct WaitcntFields(u32);
    vmcnt_lo, _: 3, 0;
    vmcnt_hi, _: 15, 14;
    expcnt, _: 6, 4;
    lgkmcnt, _: 13, 8;
}

impl WaitcntFields {
    fn from_i32(simm: i32) -> Self {
        Self(u32::from_ne_bytes(simm.to_ne_bytes()))
    }
    fn vmcnt(&self) -> u32 {
        self.vmcnt_lo() | (self.vmcnt_hi() << 4)
    }
    const VMCNT_NOOP: u32 = 0b111111;
    const EXPCNT_NOOP: u32 = 0b111;
    const LGKMCNT_NOOP: u32 = 0b111111;
}

impl Display for WaitcntFields {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let fields = [
            ("vmcnt", self.vmcnt(), Self::VMCNT_NOOP),
            ("expcnt", self.expcnt(), Self::EXPCNT_NOOP),
            ("lgkmcnt", self.lgkmcnt(), Self::LGKMCNT_NOOP),
        ];
        fields
            .iter()
            .filter(|(_, value, noop)| value != noop)
            .try_for_each(|(field, value, _)| write!(f, " {}({})", field, value))
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum MsgId {
    Interrupt = 1,
    Gs = 2,
    GsDone = 3,
    SaveWave = 4,
    StallWaveGen = 5,
    HaltWaves = 6,
    OrderedPsDone = 7,
    GsAllocReq = 9,
    GetDoorbell = 10,
    GetDdid = 11,
    SysMsg = 15,
}

impl MsgId {
    const ID_NAMES: [(MsgId, &'static str); 11] = [
        (MsgId::Interrupt, "MSG_INTERRUPT"),
        (MsgId::Gs, "MSG_GS"),
        (MsgId::GsDone, "MSG_GS_DONE"),
        (MsgId::SaveWave, "MSG_SAVEWAVE"),
        (MsgId::StallWaveGen, "MSG_STALL_WAVE_GEN"),
        (MsgId::HaltWaves, "MSG_HALT_WAVES"),
        (MsgId::OrderedPsDone, "MSG_ORDERED_PS_DONE"),
        (MsgId::GsAllocReq, "MSG_GS_ALLOC_REQ"),
        (MsgId::GetDoorbell, "MSG_GET_DOORBELL"),
        (MsgId::GetDdid, "MSG_GET_DDID"),
        (MsgId::SysMsg, "MSG_SYSMSG"),
    ];
}

impl TryFrom<u32> for MsgId {
    type Error = ();
    fn try_from(value: u32) -> std::result::Result<MsgId, ()> {
        MsgId::ID_NAMES
            .iter()
            .find(|(msg_id, _)| value == *msg_id as u32)
            .map(|(msg_id, _)| *msg_id)
            .ok_or(())
    }
}

impl From<MsgId> for &'static str {
    fn from(msg_id: MsgId) -> Self {
        MsgId::ID_NAMES
            .iter()
            .find(|(id, _)| msg_id == *id)
            .expect("Unknown MsgId")
            .1
    }
}

bitfield! {
    struct SendMsg(u32);
    id, _: 3, 0;
    operation, _: 6, 4;
    stream, _: 9, 8;
}

impl SendMsg {
    fn from_operand(o: &Operand) -> Option<Self> {
        if let Operand::Constant(imm) = o {
            Some(Self(*imm as u32))
        } else {
            None
        }
    }

    const MSG_OPERATION_STREAM: [(MsgId, u32, &'static str, bool); 10] = [
        (MsgId::Gs, 1, "GS_OP_CUT", true),
        (MsgId::Gs, 2, "GS_OP_EMIT", true),
        (MsgId::Gs, 3, "GS_OP_EMIT_CUT", true),
        (MsgId::GsDone, 0, "GS_OP_NOP", false),
        (MsgId::GsDone, 1, "GS_OP_CUT", true),
        (MsgId::GsDone, 2, "GS_OP_EMIT", true),
        (MsgId::GsDone, 3, "GS_OP_EMIT_CUT", true),
        (MsgId::SysMsg, 1, "SYSMSG_OP_ECC_ERR_INTERRUPT", false),
        (MsgId::SysMsg, 2, "SYSMSG_OP_REG_RD", false),
        (MsgId::SysMsg, 4, "SYSMSG_OP_TTRACE_PC", false),
    ];

    fn msg_operation_stream(id: MsgId, operation: u32) -> (Option<&'static str>, bool) {
        let r = Self::MSG_OPERATION_STREAM
            .iter()
            .find(|(msg_id, msg_op, _, _)| *msg_id == id && *msg_op == operation);
        match r {
            Some((_, _, operation, support_stream)) => (Some(operation), *support_stream),
            _ => (None, false),
        }
    }
}

impl Display for SendMsg {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let id = MsgId::try_from(self.id()).map_err(|_| std::fmt::Error)?;
        let operation = self.operation();
        let stream = self.stream();
        let name: &str = id.into();
        write!(f, "sendmsg({}", name)?;
        let (operation, support_stream) = Self::msg_operation_stream(id, operation);
        if let Some(operation) = operation {
            write!(f, ", {}", operation)?;
            if support_stream {
                write!(f, ", {}", stream)?;
            }
        }
        write!(f, ")")
    }
}

bitfield! {
    struct HWReg(u32);
    id, _: 5, 0;
    offset, _: 10, 6;
    size, _: 15, 11;
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum HardwareInternalRegister {
    Mode = 1,
    Status = 2,
    Trapsts = 3,
    GprAlloc = 5,
    LdsAlloc = 6,
    IbSts = 7,
    ShMemBases = 15,
    TbaLo = 16,
    TbaHi = 17,
    TmaLo = 18,
    TmaHi = 19,
    FlatScratchLo = 20,
    FlatScratchHi = 21,
    PopsPacker = 25,
    ShaderCycles = 29,
}

impl HardwareInternalRegister {
    const NAME: [(HardwareInternalRegister, &'static str); 15] = [
        (Self::Mode, "HW_REG_MODE"),
        (Self::Status, "HW_REG_STATUS"),
        (Self::Trapsts, "HW_REG_TRAPSTS"),
        (Self::GprAlloc, "HW_REG_GPR_ALLOC"),
        (Self::LdsAlloc, "HW_REG_LDS_ALLOC"),
        (Self::IbSts, "HW_REG_IB_STS"),
        (Self::ShMemBases, "HW_REG_SH_MEM_BASES"),
        (Self::TbaLo, "HW_REG_TBA_LO"),
        (Self::TbaHi, "HW_REG_TBA_HI"),
        (Self::TmaLo, "HW_REG_TMA_LO"),
        (Self::TmaHi, "HW_REG_TMA_HI"),
        (Self::FlatScratchLo, "HW_REG_FLAT_SCR_LO"),
        (Self::FlatScratchHi, "HW_REG_FLAT_SCR_HI"),
        (Self::PopsPacker, "HW_REG_POPS_PACKER"),
        (Self::ShaderCycles, "HW_REG_SHADER_CYCLES"),
    ];
}

impl From<HardwareInternalRegister> for &'static str {
    fn from(r: HardwareInternalRegister) -> Self {
        HardwareInternalRegister::NAME
            .iter()
            .find(|(reg, _)| *reg == r)
            .expect("invalid hardware register")
            .1
    }
}

impl TryFrom<u32> for HardwareInternalRegister {
    type Error = ();
    fn try_from(value: u32) -> std::result::Result<Self, ()> {
        let r = HardwareInternalRegister::NAME
            .iter()
            .find(|(reg, _)| *reg as u32 == value)
            .ok_or(())?;
        Ok(r.0)
    }
}

impl HWReg {
    const SIZE_DEFAULT: u32 = 32;
    const OFFSET_DEFAULT: u32 = 0;

    fn from_operand(o: &Operand) -> Option<Self> {
        if let Operand::Constant(imm) = o {
            Some(Self(*imm as u32))
        } else {
            None
        }
    }
}

impl Display for HWReg {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let reg = HardwareInternalRegister::try_from(self.id()).map_err(|_| std::fmt::Error)?;
        let name: &str = reg.into();
        let offset = self.offset();
        let size = self.size() + 1;
        if offset != Self::OFFSET_DEFAULT || size != Self::SIZE_DEFAULT {
            write!(f, "hwreg({}, {}, {})", name, offset, size)
        } else {
            write!(f, "hwreg({})", name)
        }
    }
}

pub(crate) enum SOPPInstExpectedOperands {
    NoOperand,
    ReadSimm,
    ConditionalBranch(Operand),
}

impl SOPPInstExpectedOperands {
    pub(crate) fn from_opcode(opcode: SOPPOpcode) -> Self {
        use SOPPInstExpectedOperands::*;
        use SOPPOpcode::*;
        //
        // XXX: The implementation does not classify CBRANCH_CDBG* as conditional branch for now
        // because they are not really used and the ISA disallows specifying the hardware status register as operand.
        match opcode {
            S_BARRIER
            | S_CODE_END
            | S_ENDPGM
            | S_ENDPGM_ORDERED_PS_DONE
            | S_ENDPGM_SAVED
            | S_ICACHE_INV
            | S_TTRACEDATA
            | S_WAKEUP => NoOperand,
            S_CBRANCH_SCC0 | S_CBRANCH_SCC1 => {
                ConditionalBranch(Operand::SpecialScalarRegister(Operand::SPECIAL_REG_SCC))
            }
            S_CBRANCH_VCCZ | S_CBRANCH_VCCNZ => {
                ConditionalBranch(Operand::SpecialScalarRegister(Operand::SPECIAL_REG_VCC_LO))
            }
            S_CBRANCH_EXECZ | S_CBRANCH_EXECNZ => {
                ConditionalBranch(Operand::SpecialScalarRegister(Operand::SPECIAL_REG_EXEC_LO))
            }
            _ => ReadSimm,
        }
    }
}

bitflags! {
    struct VOPCOpcodeFlags : u8 {
        const OPERATION = 0b00000111;
        const WRITE_EXEC = 0b00010000;
        const TYPE = 0b11101000;
    }
}

impl VOPCOpcodeFlags {
    pub fn from_opcode(opcode: VOPCOpcode) -> Self {
        VOPCOpcodeFlags::from_bits_truncate(opcode as u8)
    }
    pub fn write_exec(&self) -> bool {
        self.contains(VOPCOpcodeFlags::WRITE_EXEC)
    }
    // vopc instructions types
    // V_CMP*_CLASS_F* instructions cannot be classified into the following types
    const V_CMP_OP_F32: u8 = 0x00;
    const V_CMP_NOP_F32: u8 = 0x08;
    const V_CMP_OP_F64: u8 = 0x20;
    const V_CMP_NOP_F64: u8 = 0x28;
    const V_CMP_OP_I32: u8 = 0x80;
    const V_CMP_OP_I16: u8 = 0x88;
    const V_CMP_OP_I64: u8 = 0xA0;
    const V_CMP_OP_U16: u8 = 0xA8;
    const V_CMP_OP_U32: u8 = 0xC0;
    const V_CMP_OP_F16: u8 = 0xC8;
    const V_CMP_OP_U64: u8 = 0xE0;
    const V_CMP_NOP_F16: u8 = 0xE8;
}

pub(crate) enum CmpDataSize {
    B16,
    B32,
    B64,
}

pub(crate) enum CmpInstType {
    UnsignedComparison,
    SignedComparison,
    FloatComparion,
    BitTest,
    FloatClass,
}

impl CmpInstType {
    fn src_size(&self, data_size: CmpDataSize) -> (usize, usize) {
        let s0_size = match data_size {
            CmpDataSize::B16 | CmpDataSize::B32 => 1,
            CmpDataSize::B64 => 2,
        };
        let s1_size = match self {
            CmpInstType::BitTest | CmpInstType::FloatClass => 1,
            _ => s0_size,
        };
        (s0_size, s1_size)
    }
}

impl VOPCOpcode {
    pub(crate) fn get_type_size(&self) -> Option<(CmpInstType, CmpDataSize)> {
        use CmpDataSize::*;
        use CmpInstType::*;
        use VOPCOpcode::*;
        match self {
            // V_CMP*_CLASS_F* instructions can not be classified by
            // `VOPCOpcodeFlags::TYPE` bits
            V_CMP_CLASS_F16 | V_CMPX_CLASS_F16 => Some((FloatClass, B16)),
            V_CMP_CLASS_F32 | V_CMPX_CLASS_F32 => Some((FloatClass, B32)),
            V_CMP_CLASS_F64 | V_CMPX_CLASS_F64 => Some((FloatClass, B64)),
            _ => match (*self as u8) & VOPCOpcodeFlags::TYPE.bits() {
                VOPCOpcodeFlags::V_CMP_OP_F16 | VOPCOpcodeFlags::V_CMP_NOP_F16 => {
                    Some((FloatComparion, B16))
                }
                VOPCOpcodeFlags::V_CMP_OP_F32 | VOPCOpcodeFlags::V_CMP_NOP_F32 => {
                    Some((FloatComparion, B32))
                }
                VOPCOpcodeFlags::V_CMP_OP_F64 | VOPCOpcodeFlags::V_CMP_NOP_F64 => {
                    Some((FloatComparion, B64))
                }
                VOPCOpcodeFlags::V_CMP_OP_I16 => Some((SignedComparison, B16)),
                VOPCOpcodeFlags::V_CMP_OP_I32 => Some((SignedComparison, B32)),
                VOPCOpcodeFlags::V_CMP_OP_I64 => Some((SignedComparison, B64)),
                VOPCOpcodeFlags::V_CMP_OP_U16 => Some((UnsignedComparison, B16)),
                VOPCOpcodeFlags::V_CMP_OP_U32 => Some((UnsignedComparison, B32)),
                VOPCOpcodeFlags::V_CMP_OP_U64 => Some((UnsignedComparison, B64)),
                _ => unreachable!("Invalid decoding for VOPC instruction"),
            },
        }
    }
}

impl SOPCOpcode {
    fn get_type_size(&self) -> (CmpInstType, CmpDataSize) {
        use CmpDataSize::*;
        use CmpInstType::*;
        use SOPCOpcode::*;
        match self {
            S_BITCMP0_B32 | S_BITCMP1_B32 => (BitTest, B32),
            S_BITCMP0_B64 | S_BITCMP1_B64 => (BitTest, B64),
            S_CMP_EQ_I32 | S_CMP_LG_I32 | S_CMP_GT_I32 | S_CMP_GE_I32 | S_CMP_LT_I32
            | S_CMP_LE_I32 => (SignedComparison, B32),
            S_CMP_EQ_U32 | S_CMP_LG_U32 | S_CMP_GT_U32 | S_CMP_GE_U32 | S_CMP_LT_U32
            | S_CMP_LE_U32 => (UnsignedComparison, B32),
            S_CMP_EQ_U64 | S_CMP_LG_U64 => (UnsignedComparison, B64),
        }
    }
}

enum SrcModifier {
    Sext,
    Neg,
    Abs,
    SextNeg,
    SextAbs,
    NegAbs,
    SextNegAbs,
}

impl TryFrom<u32> for SrcModifier {
    type Error = ();
    fn try_from(value: u32) -> std::result::Result<SrcModifier, ()> {
        use SrcModifier::*;
        match value {
            0b001 => Ok(Sext),
            0b010 => Ok(Neg),
            0b100 => Ok(Abs),
            0b011 => Ok(SextNeg),
            0b101 => Ok(SextAbs),
            0b110 => Ok(NegAbs),
            0b111 => Ok(SextNegAbs),
            _ => Err(()),
        }
    }
}

impl Operand {
    fn modified_by(&self, modifier: Option<SrcModifier>) -> impl Display + '_ {
        ModifiedOperand {
            modifier,
            operand: self,
        }
    }
}

struct FormattedOperand<'a> {
    operand: &'a Operand,
    precision: usize,
    alternate: bool,
}
impl<'a> Display for FormattedOperand<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.alternate {
            write!(f, "{:#.*}", self.precision, self.operand)
        } else {
            write!(f, "{:.*}", self.precision, self.operand)
        }
    }
}

struct ModifiedOperand<'a> {
    modifier: Option<SrcModifier>,
    operand: &'a Operand,
}
impl<'a> Display for ModifiedOperand<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use SrcModifier::*;
        let operand = FormattedOperand {
            operand: self.operand,
            precision: f.precision().unwrap_or(1),
            alternate: f.alternate(),
        };
        match self.modifier {
            None => write!(f, "{}", operand),
            Some(Sext) => write!(f, "sext({})", operand),
            Some(Neg) => write!(f, "-{}", operand),
            Some(Abs) => write!(f, "|{}|", operand),
            Some(SextNeg) => write!(f, "sext(-{})", operand),
            Some(SextAbs) => write!(f, "sext(|{}|)", operand),
            Some(NegAbs) => write!(f, "-|{}|", operand),
            Some(SextNegAbs) => write!(f, "sext(-|{}|)", operand),
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum OMod {
    Mul2,
    Mul4,
    Div2,
}

impl TryFrom<u32> for OMod {
    type Error = ();
    fn try_from(value: u32) -> std::result::Result<OMod, ()> {
        match value {
            1 => Ok(OMod::Mul2),
            2 => Ok(OMod::Mul4),
            3 => Ok(OMod::Div2),
            _ => Err(()),
        }
    }
}

impl Display for OMod {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            OMod::Mul2 => f.pad("mul:2"),
            OMod::Mul4 => f.pad("mul:4"),
            OMod::Div2 => f.pad("div:2"),
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum SDWAOperandSel {
    Byte0,
    Byte1,
    Byte2,
    Byte3,
    Word0,
    Word1,
    DWord,
}

impl TryFrom<u32> for SDWAOperandSel {
    type Error = ();

    fn try_from(value: u32) -> std::result::Result<SDWAOperandSel, ()> {
        use SDWAOperandSel::*;
        match value {
            0 => Ok(Byte0),
            1 => Ok(Byte1),
            2 => Ok(Byte2),
            3 => Ok(Byte3),
            4 => Ok(Word0),
            5 => Ok(Word1),
            6 => Ok(DWord),
            _ => Err(()),
        }
    }
}

impl Display for SDWAOperandSel {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use SDWAOperandSel::*;
        match self {
            Byte0 => f.pad("BYTE_0"),
            Byte1 => f.pad("BYTE_1"),
            Byte2 => f.pad("BYTE_2"),
            Byte3 => f.pad("BYTE_3"),
            Word0 => f.pad("WORD_0"),
            Word1 => f.pad("WORD_1"),
            DWord => f.pad("DWORD"),
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum DstUnused {
    Pad,
    Sext,
    Preserve,
}

impl TryFrom<u32> for DstUnused {
    type Error = ();

    fn try_from(value: u32) -> std::result::Result<DstUnused, ()> {
        use DstUnused::*;
        match value {
            0 => Ok(Pad),
            1 => Ok(Sext),
            2 => Ok(Preserve),
            _ => Err(()),
        }
    }
}

impl Display for DstUnused {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use DstUnused::*;
        match self {
            Pad => f.pad("UNUSED_PAD"),
            Sext => f.pad("UNUSED_SEXT"),
            Preserve => f.pad("UNUSED_PRESERVE"),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum InstructionModifier {
    Sdwa(SDWAModifier),
    Dpp16(DPP16Modifier),
    Vop3(VOP3Modifier),
    Vop3p(VOP3PModifier),
    Ds(DSModifier),
    VMem(VMEMModifier),
    Mubuf(MUBUFModifier),
}

bitfield! {
    #[derive(Clone, Copy, Debug)]
    pub struct DSModifier(u32);
    offset, _: 15, 0;
    offset0, _: 7, 0;
    offset1, _: 15, 8;
    gds, _: 17, 17;
}

impl DSModifier {
    fn fmt(&self, f: &mut Formatter<'_>, offset_count: u8) -> std::fmt::Result {
        let offsets: SmallVec<[_; 2]> = match offset_count {
            0 => smallvec![],
            1 => smallvec![("offset", self.offset())],
            2 => smallvec![("offset0", self.offset0()), ("offset1", self.offset1())],
            _ => Err(std::fmt::Error)?,
        };
        offsets
            .iter()
            .filter(|(_, value)| *value != 0)
            .try_for_each(|(name, value)| write!(f, " {}:{}", name, value))?;
        if self.gds() != 0 {
            write!(f, " gds")?;
        }
        Ok(())
    }
}

bitfield! {
    #[derive(Clone, Copy)]
    pub struct SDWADstModifier(u32);
    dst_sel, _: 2, 0;
    dst_u, _: 4, 3;
    clmp, _: 5, 5;
    omod, _: 7, 6;
}

impl Display for SDWADstModifier {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let omod = OMod::try_from(self.omod()).ok();
        let dst_sel = SDWAOperandSel::try_from(self.dst_sel()).map_err(|_| std::fmt::Error)?;
        let dst_u = DstUnused::try_from(self.dst_u()).map_err(|_| std::fmt::Error)?;
        if self.clmp() != 0 {
            write!(f, "clamp")?;
        }
        if let Some(omod) = omod {
            write!(f, " {}", omod)?;
        }
        write!(f, " dst_sel:{} dst_unused:{}", dst_sel, dst_u)
    }
}

bitfield! {
    #[derive(Clone, Copy)]
    pub struct SDWABDstModifier(u32);
    sdst, _: 6, 0;
    sd, _: 7, 7;
}

bitfield! {
    #[derive(Clone, Copy, Debug)]
    pub struct SDWAModifier(u32);
    src0, _: 7, 0;
    dst_modifier, _: 15, 8;
    src0_sel, _: 18, 16;
    src0_modifier, _: 21, 19;
    src0_sgpr, _: 23, 23;
    src1_sel, _: 26, 24;
    src1_modifier, _: 29, 27;
    src1_sgpr, _: 31, 31;
}

impl SDWAModifier {
    fn parse_src(src: u8, is_scalar: bool) -> Result<Operand> {
        if is_scalar {
            if src <= Operand::MAX_SGPR_NUM as u8 {
                Ok(Operand::ScalarRegister(src, 1))
            } else {
                Err(Error::DecodeError(DecodeError::InvalidOpcode))
            }
        } else {
            Ok(Operand::VectorRegister(src, 1))
        }
    }
    fn parse_src0(&self) -> Result<Operand> {
        Self::parse_src(self.src0() as u8, self.src0_sgpr() != 0)
    }
    fn parse_src1(&self, src1: u8) -> Result<Operand> {
        Self::parse_src(src1, self.src1_sgpr() != 0)
    }
    fn parse_sdwab_dst(&self, default: u8) -> Result<Operand> {
        let dst_modifier = SDWABDstModifier(self.dst_modifier());
        if dst_modifier.sd() != 0 {
            Self::parse_src(dst_modifier.sdst() as u8, true)
        } else {
            Ok(Operand::SpecialScalarRegister(default))
        }
    }
    fn parse_src0_modifier(&self) -> Option<SrcModifier> {
        SrcModifier::try_from(self.src0_modifier()).ok()
    }
    fn parse_src1_modifier(&self) -> Option<SrcModifier> {
        SrcModifier::try_from(self.src1_modifier()).ok()
    }
    fn fmt_sdwa_dst_modifier(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let dst_m = SDWADstModifier(self.dst_modifier());
        write!(f, "{}", dst_m)
    }
    fn fmt_src0_sel(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let sel = SDWAOperandSel::try_from(self.src0_sel()).map_err(|_| std::fmt::Error)?;
        write!(f, " src0_sel:{}", sel)
    }
    fn fmt_src1_sel(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let sel = SDWAOperandSel::try_from(self.src1_sel()).map_err(|_| std::fmt::Error)?;
        write!(f, " src1_sel:{}", sel)
    }
}

bitfield! {
    /// For VOP3A/B instructions
    #[derive(Clone, Debug)]
    pub struct VOP3Modifier(u32);
    abs, _: 10, 8;
    op_sel, _: 14, 11;
    clmp, _: 15, 15;
    omod, _: 28, 27;
    neg, _: 31, 29;
}

impl VOP3Modifier {
    const HI_MASK: u32 = 0xf8000000;
    const VOP3A_LO_MASK: u32 = 0x0000ff00;
    const VOP3B_LO_MASK: u32 = 0x00008000;
    fn from_vop3a_inst(lo: u32, hi: u32) -> Self {
        Self((lo & Self::VOP3A_LO_MASK) | (hi & Self::HI_MASK))
    }
    fn from_vop3b_inst(lo: u32, hi: u32) -> Self {
        Self((lo & Self::VOP3B_LO_MASK) | (hi & Self::HI_MASK))
    }
    fn parse_src_modifier(&self, op_idx: usize) -> Option<SrcModifier> {
        match (
            ((self.abs() >> op_idx) & 1) != 0,
            ((self.neg() >> op_idx) & 1) != 0,
        ) {
            (false, false) => None,
            (true, false) => Some(SrcModifier::Abs),
            (false, true) => Some(SrcModifier::Neg),
            (true, true) => Some(SrcModifier::NegAbs),
        }
    }
    fn fmt_vop3b_dst_modifier(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.clmp() != 0 {
            write!(f, " clamp")?;
        }
        if let Ok(omod) = OMod::try_from(self.omod()) {
            write!(f, " {}", omod)?;
        }
        Ok(())
    }
    fn fmt_vop3a_dst_modifier(&self, f: &mut Formatter<'_>, src_count: usize) -> std::fmt::Result {
        let op_sel = self.op_sel();
        if op_sel != 0 {
            let bits: SmallVec<[u32; 4]> = (0..4).map(|i| (op_sel >> i) & 1).collect();
            write!(f, " op_sel:[")?;
            bits.iter()
                .take(src_count)
                .try_for_each(|x| write!(f, "{},", x))?;
            write!(f, "{}", bits[3])?; // op_sel for dst
            write!(f, "]")?;
        }
        if self.clmp() != 0 {
            write!(f, " clamp")?;
        }
        if let Ok(omod) = OMod::try_from(self.omod()) {
            write!(f, " {}", omod)?;
        }
        Ok(())
    }
}

bitfield! {
    #[derive(Clone, Debug)]
    pub struct VOP3PModifier(u32);
    neg_hi, _: 10, 8;
    op_sel, _: 13, 11;
    op_sel_hi_2, _: 14, 14;
    clmp, _: 15, 15;
    op_sel_hi_01, _: 28, 27;
    neg, _: 31, 29;
}

impl VOP3PModifier {
    const HI_MASK: u32 = 0xf8000000;
    const LO_MASK: u32 = 0x0000ff00;
    fn from_inst(lo: u32, hi: u32) -> Self {
        Self((lo & Self::LO_MASK) | (hi & Self::HI_MASK))
    }
    fn fmt_packed_modifier(
        f: &mut Formatter<'_>,
        name: &str,
        bits: u32,
        default: u32,
        src_count: usize,
    ) -> std::fmt::Result {
        let is_default = (0..src_count).all(|i| (bits >> i) & 1 == default);
        if !is_default {
            write!(f, " {}:[{}", name, bits & 1)?;
            (1..src_count).try_for_each(|i| write!(f, ",{}", (bits >> i) & 1))?;
            write!(f, "]")?;
        }
        Ok(())
    }
    fn fmt_modifier(&self, f: &mut Formatter<'_>, opcode: VOP3POpcode) -> std::fmt::Result {
        let src_count = opcode.src_count();
        if opcode.should_print_op_sel() {
            Self::fmt_packed_modifier(f, "op_sel", self.op_sel(), 0, src_count)?;
            let op_sel_hi = self.op_sel_hi_2() << 2 | self.op_sel_hi_01();
            Self::fmt_packed_modifier(f, "op_sel_hi", op_sel_hi, 1, src_count)?;
        }
        if opcode.should_print_neg() {
            Self::fmt_packed_modifier(f, "neg_lo", self.neg(), 0, src_count)?;
            Self::fmt_packed_modifier(f, "neg_hi", self.neg_hi(), 0, src_count)?;
        }
        if self.clmp() != 0 {
            write!(f, " clamp")?;
        }
        Ok(())
    }
}

enum DPP16Ctrl {
    QuadPerm([u8; 4]),
    RowMirror,
    RowHalfMirror,
    RowShl(u32),
    RowShr(u32),
    RowRor(u32),
}

impl DPP16Ctrl {
    const QUAD_PERM_MIN: u32 = 0x000;
    const QUAD_PERM_MAX: u32 = 0x0ff;
    const QUAD_PERM_FIELD_MASK: u32 = 0x3;
    const ROW_SHL_MIN: u32 = 0x100;
    const ROW_SHL_MAX: u32 = 0x10f;
    const ROW_SHR_MIN: u32 = 0x110;
    const ROW_SHR_MAX: u32 = 0x11f;
    const ROW_ROR_MIN: u32 = 0x120;
    const ROW_ROR_MAX: u32 = 0x12f;
    const ROW_SHIFT_MASK: u32 = 0xf;
    const ROW_MIRROR: u32 = 0x140;
    const ROW_HALF_MIRROR: u32 = 0x141;
}

impl Display for DPP16Ctrl {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use DPP16Ctrl::*;
        match self {
            QuadPerm([a, b, c, d]) => write!(f, "quad_perm:[{},{},{},{}]", a, b, c, d),
            RowMirror => write!(f, "row_mirror"),
            RowHalfMirror => write!(f, "row_half_mirror"),
            RowShl(x) => write!(f, "row_shl:{}", x),
            RowShr(x) => write!(f, "row_shr:{}", x),
            RowRor(x) => write!(f, "row_ror:{}", x),
        }
    }
}

impl TryFrom<u32> for DPP16Ctrl {
    type Error = ();
    fn try_from(value: u32) -> std::result::Result<DPP16Ctrl, ()> {
        use DPP16Ctrl::*;
        match value {
            x if (Self::QUAD_PERM_MIN..=Self::QUAD_PERM_MAX).contains(&x) => {
                let offset = [0, 2, 4, 6];
                let perm =
                    offset.map(|offset| ((value >> offset) & Self::QUAD_PERM_FIELD_MASK) as u8);
                Ok(QuadPerm(perm))
            }
            x if (Self::ROW_SHL_MIN..=Self::ROW_SHL_MAX).contains(&x) => {
                Ok(RowShl(value & Self::ROW_SHIFT_MASK))
            }
            x if (Self::ROW_SHR_MIN..=Self::ROW_SHR_MAX).contains(&x) => {
                Ok(RowShr(value & Self::ROW_SHIFT_MASK))
            }
            x if (Self::ROW_ROR_MIN..=Self::ROW_ROR_MAX).contains(&x) => {
                Ok(RowRor(value & Self::ROW_SHIFT_MASK))
            }
            Self::ROW_MIRROR => Ok(RowMirror),
            Self::ROW_HALF_MIRROR => Ok(RowHalfMirror),
            _ => Err(()),
        }
    }
}

bitfield! {
    #[derive(Clone, Copy, Debug)]
    pub struct DPP16Modifier(u32);
    src0, _: 7, 0;
    dpp_ctrl, _: 16, 8;
    fi, _: 18, 18;
    bc, _: 19, 19;
    src0_neg, _: 20, 20;
    src0_abs, _: 21, 21;
    src1_neg, _: 22, 22;
    src1_abs, _: 23, 23;
    bank_mask, _: 27, 24;
    row_mask, _: 31, 28;
}

impl DPP16Modifier {
    fn parse_src0(&self) -> Operand {
        Operand::VectorRegister(self.src0() as u8, 1)
    }
    fn parse_src_modifier(neg: bool, abs: bool) -> Option<SrcModifier> {
        use SrcModifier::*;
        match (neg, abs) {
            (false, false) => None,
            (false, true) => Some(Abs),
            (true, false) => Some(Neg),
            (true, true) => Some(NegAbs),
        }
    }
    fn parse_src0_modifier(&self) -> Option<SrcModifier> {
        Self::parse_src_modifier(self.src0_neg() != 0, self.src0_abs() != 0)
    }
    fn parse_src1_modifier(&self) -> Option<SrcModifier> {
        Self::parse_src_modifier(self.src1_neg() != 0, self.src1_abs() != 0)
    }
}

impl Display for DPP16Modifier {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let dpp_ctrl = DPP16Ctrl::try_from(self.dpp_ctrl()).map_err(|_| std::fmt::Error)?;
        write!(f, "{}", dpp_ctrl)?;
        write!(f, " row_mask:{:#x}", self.row_mask())?;
        write!(f, " bank_mask:{:#x}", self.bank_mask())?;
        if self.bc() == 1 {
            // Both bound_ctrl:0 and bound_ctrl:1 are encoded as 1
            // See
            // - Bug report: https://bugs.llvm.org/show_bug.cgi?id=35397
            // - LLVM's parser: https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AsmParser/AMDGPUAsmParser.cpp#L8039
            // - disassembler in LLVM-12 (prints " bound_ctrl:0"): https://github.com/llvm/llvm-project/blob/release/12.x/llvm/lib/Target/AMDGPU/MCTargetDesc/AMDGPUInstPrinter.cpp#L894
            // - disassembler in LLVM-latest (prints " bound_ctrl:1"): https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/MCTargetDesc/AMDGPUInstPrinter.cpp#L1021
            f.pad(" bound_ctrl:0")?;
        }
        if self.fi() == 1 {
            f.pad(" fi:1")?;
        }
        Ok(())
    }
}

impl VOP3ABOpcode {
    pub(crate) fn src_count(&self) -> usize {
        use VOP3ABInstType::*;
        use VOP3ABOpcode::*;
        match VOP3ABInstType::from_opcode(*self) {
            VOP3Specific => match self {
                V_LDEXP_F32 | V_LDEXP_F64 => 1,
                V_ADD_CO_U32 | V_ADD_F64 | V_ADD_NC_I32 | V_ADD_NC_U16 | V_ADD_NC_I16
                | V_ASHRREV_I16 | V_ASHRREV_I64 | V_BCNT_U32_B32 | V_BFM_B32 | V_CVT_PK_I16_I32
                | V_CVT_PK_U16_U32 | V_CVT_PKNORM_I16_F16 | V_CVT_PKNORM_U16_F16
                | V_CVT_PKNORM_I16_F32 | V_CVT_PKNORM_U16_F32 | V_INTERP_P1LL_F16
                | V_INTERP_P1LV_F16 | V_INTERP_P2_F16 | V_LSHLREV_B16 | V_LSHLREV_B64
                | V_LSHRREV_B16 | V_LSHRREV_B64 | V_MAX_U16 | V_MAX_I16 | V_MAX_F64
                | V_MBCNT_HI_U32_B32 | V_MBCNT_LO_U32_B32 | V_MIN_U16 | V_MIN_I16 | V_MIN_F64
                | V_MUL_F64 | V_MUL_HI_I32 | V_MUL_HI_U32 | V_MUL_LO_U16 | V_MUL_LO_U32
                | V_PACK_B32_F16 | V_READLANE_B32 | V_SUB_CO_U32 | V_SUB_NC_I32 | V_SUB_NC_U16
                | V_SUB_NC_I16 | V_SUBREV_CO_U32 | V_WRITELANE_B32 => 2,
                _ => 3,
            },
            FromVOP1 => match self {
                V_CLREXCP | V_PIPEFLUSH | V_NOP => 0,
                _ => 1,
            },
            FromVOP2 => match self {
                V_CNDMASK_B32 | V_ADD_CO_CI_U32 | V_SUB_CO_CI_U32 | V_SUBREV_CO_CI_U32 => 3,
                _ => 2,
            },
            FromVOPC => 2,
        }
    }

    /// register size for vdst, src0, src1, src2
    fn reg_size(&self) -> (u8, u8, u8, u8) {
        use VOP3ABInstType::*;
        use VOP3ABOpcode::*;
        match VOP3ABInstType::from_opcode(*self) {
            VOP3Specific => match self {
                V_MAD_U64_U32 | V_MAD_I64_I32 => (2, 1, 1, 2),
                V_ASHRREV_I64 | V_LSHLREV_B64 | V_LSHRREV_B64 => (2, 1, 2, 0),
                V_ADD_F64 | V_CEIL_F64 | V_DIV_FIXUP_F64 | V_DIV_SCALE_F64 | V_DIV_FMAS_F64
                | V_FMA_F64 | V_MAX_F64 | V_MUL_F64 => (2, 2, 2, 2),
                _ => (1, 1, 1, 1),
            },
            FromVOP1 => match self {
                V_CEIL_F64 | V_FLOOR_F64 | V_FRACT_F64 | V_FREXP_MANT_F64 | V_RCP_F64
                | V_RNDNE_F64 | V_RSQ_F64 | V_SQRT_F64 | V_TRUNC_F64 => (2, 2, 0, 0),
                V_CVT_F32_F64 | V_CVT_I32_F64 | V_CVT_U32_F64 | V_FREXP_EXP_I32_F64 => (1, 2, 0, 0),
                V_CVT_F64_F32 | V_CVT_F64_I32 | V_CVT_F64_U32 => (2, 1, 0, 0),
                _ => (1, 1, 0, 0),
            },
            FromVOP2 => (1, 1, 1, 1),
            FromVOPC => {
                let vopc = self.as_vopc().unwrap();
                let (inst_type, data_size) = vopc.get_type_size().unwrap();
                let (s0_size, s1_size) = inst_type.src_size(data_size);
                (1, s0_size as u8, s1_size as u8, 0)
            }
        }
    }

    /// print opcode with suffix `_e64`
    fn explicit_64bit_encoding(&self) -> bool {
        !matches!(
            VOP3ABInstType::from_opcode(*self),
            VOP3ABInstType::VOP3Specific
        )
    }
}

pub(crate) enum VOP3ABInstType {
    VOP3Specific,
    FromVOPC,
    FromVOP1,
    FromVOP2,
}

impl VOP3ABInstType {
    pub(crate) fn from_opcode(opcode: VOP3ABOpcode) -> Self {
        use VOP3ABInstType::*;
        match opcode {
            o if VOP3ABOpcode::VOP1_AS_VOP3_MIN <= o && o <= VOP3ABOpcode::VOP1_AS_VOP3_MAX => {
                FromVOP1
            }
            o if VOP3ABOpcode::VOP2_AS_VOP3_MIN <= o && o <= VOP3ABOpcode::VOP2_AS_VOP3_MAX => {
                FromVOP2
            }
            o if VOP3ABOpcode::VOPC_AS_VOP3_MIN <= o && o <= VOP3ABOpcode::VOPC_AS_VOP3_MAX => {
                FromVOPC
            }
            _ => VOP3Specific,
        }
    }
}

pub(crate) enum VOP3ABDstType {
    NoDst,
    VDst,
    VDstSDst,
    VDstAsSGPR,
    Exec,
}

impl VOP3ABDstType {
    pub(crate) fn from_opcode(opcode: VOP3ABOpcode) -> Self {
        use VOP3ABDstType::*;
        use VOP3ABInstType::*;
        use VOP3ABOpcode::*;
        if opcode.is_vop3b() {
            VDstSDst
        } else {
            match VOP3ABInstType::from_opcode(opcode) {
                FromVOPC => {
                    let vopc = opcode.as_vopc().unwrap();
                    if VOPCOpcodeFlags::from_opcode(vopc).write_exec() {
                        Exec
                    } else {
                        VDstAsSGPR
                    }
                }
                _ => match opcode {
                    V_CLREXCP | V_PIPEFLUSH => NoDst,
                    V_READLANE_B32 => VDstAsSGPR,
                    _ => VDst,
                },
            }
        }
    }
}

impl VOP3ABOpcode {
    const VOP1_AS_VOP3_MIN: VOP3ABOpcode = VOP3ABOpcode::V_NOP;
    const VOP1_AS_VOP3_MAX: VOP3ABOpcode = VOP3ABOpcode::V_SWAPREL_B32;
    const VOP2_AS_VOP3_MIN: VOP3ABOpcode = VOP3ABOpcode::V_CNDMASK_B32;
    const VOP2_AS_VOP3_MAX: VOP3ABOpcode = VOP3ABOpcode::V_PK_FMAC_F16;
    const VOPC_AS_VOP3_MIN: VOP3ABOpcode = VOP3ABOpcode::V_CMP_F_F32;
    const VOPC_AS_VOP3_MAX: VOP3ABOpcode = VOP3ABOpcode::V_CMPX_TRU_F16;
    pub(crate) fn as_vopc(&self) -> Option<VOPCOpcode> {
        VOPCOpcode::try_from(*self as u32).ok()
    }
}

impl VOP3POpcode {
    pub(crate) fn src_count(&self) -> usize {
        use VOP3POpcode::*;
        match self {
            V_DOT2_F32_F16 | V_DOT2_I32_I16 | V_DOT2_U32_U16 | V_DOT4_I32_I8 | V_DOT4_U32_U8
            | V_DOT8_I32_I4 | V_DOT8_U32_U4 | V_FMA_MIX_F32 | V_FMA_MIXHI_F16 | V_FMA_MIXLO_F16
            | V_PK_FMA_F16 | V_PK_MAD_I16 | V_PK_MAD_U16 => 3,
            _ => 2,
        }
    }
    fn should_print_op_sel(&self) -> bool {
        use VOP3POpcode::*;
        !matches!(
            self,
            V_DOT4_I32_I8 | V_DOT4_U32_U8 | V_DOT8_I32_I4 | V_DOT8_U32_U4
        )
    }
    fn should_print_neg(&self) -> bool {
        use VOP3POpcode::*;
        matches!(
            self,
            V_DOT2_F32_F16
                | V_PK_ADD_F16
                | V_PK_FMA_F16
                | V_PK_MAX_F16
                | V_PK_MIN_F16
                | V_PK_MUL_F16
        )
    }
}

pub struct Decoder<'a> {
    data: &'a [u32],
    offset: usize,
    // An instruction might optionally use a 32-bit literal constant that immediately follows the instruction
    literal: Option<u32>,
}

pub enum DecodeOperand {
    // The lo / hi bits and the size of the group
    ScalarDst(u8, u8, u8),
    ScalarSrc(u8, u8, u8),
    ScalarImm(u32, u32),
    ScalarSignedImm(u32, u32),
    VectorSrc(u8, u8, u8),
    VectorReg(u8, u8, u8),
    SpecialScalarReg(u8),
    // common sgpr with LSBs filled with 0, e.g.
    // - `SBASE` in Table 73 has a size of 2 or 4 and missed 1 LSB;
    // - `SRSRC` in Table 98 has a size of 4 and missed 2 LSB's.
    // the lo / hi bits, the size and count of LSBs
    ScalarRegSlice(u8, u8, u8, u8),
    Literal,
    // modifier, default register (VCC or EXEC depends on the opcode) when sd = 0
    SDWABDst(SDWAModifier, u8),
    // modifier
    SDWASrc0(SDWAModifier),
    // modifier, lo, hi
    SDWASrc1(SDWAModifier, u8, u8),
    Dpp16Src0(DPP16Modifier),
    Void,
}

#[derive(Copy, Clone)]
struct DecodeOption(u32);

impl DecodeOption {
    const CONSTANT: u32 = 1;
    const LITERAL: u32 = 2;
    const VREG: u32 = 4;
    const SCALAR_REG_SLICE: u32 = 8;
    const REG_LEN_SHIFT: u32 = 4;
    fn new() -> DecodeOption {
        DecodeOption(1 << Self::REG_LEN_SHIFT)
    }
    fn constant(self) -> DecodeOption {
        DecodeOption(self.0 | Self::CONSTANT)
    }
    fn literal(self) -> DecodeOption {
        DecodeOption(self.0 | Self::LITERAL)
    }
    fn vreg(self) -> DecodeOption {
        DecodeOption(self.0 | Self::VREG)
    }
    fn scalar_reg_slice(self, l: u8) -> DecodeOption {
        let v = (self.0 | Self::SCALAR_REG_SLICE) & ((1 << Self::REG_LEN_SHIFT) - 1);
        DecodeOption(v | ((l as u32) << Self::REG_LEN_SHIFT))
    }
    fn reg_len(self, l: u8) -> DecodeOption {
        let v = self.0 & ((1 << Self::REG_LEN_SHIFT) - 1);
        DecodeOption(v | ((l as u32) << Self::REG_LEN_SHIFT))
    }
    fn get_reg_len(&self) -> u8 {
        (self.0 >> Self::REG_LEN_SHIFT) as u8
    }
    fn contains(&self, flag: u32) -> bool {
        (self.0 & flag) != 0
    }
}

impl<'a> Decoder<'a> {
    pub fn new(data: &'a [u32]) -> Decoder<'a> {
        Decoder {
            data,
            offset: 0,
            literal: None,
        }
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    fn read(&mut self) -> Result<u32> {
        if self.offset >= self.data.len() {
            return Err(Error::IOError(io::Error::from(ErrorKind::UnexpectedEof)));
        }
        let res = self.data[self.offset];
        self.offset += 1;
        Ok(res)
    }

    fn get_or_read_literal(&mut self) -> Result<u32> {
        Ok(match self.literal {
            Some(x) => x,
            None => {
                let v = self.read()?;
                self.literal = Some(v);
                v
            }
        })
    }

    fn decode_some(&mut self) -> Result<Instruction> {
        let lo = self.read()?;
        let op = Opcode::from(lo);
        self.literal = None;
        match op {
            Opcode::SOP2(opcode) => self.parse_sop2_inst(op, lo, opcode),
            Opcode::SOPK(opcode) => self.parse_sopk_inst(op, lo, opcode),
            Opcode::SOP1(opcode) => self.parse_sop1_inst(op, lo, opcode),
            Opcode::SOPC(opcode) => self.parse_sopc_inst(op, lo, opcode),
            Opcode::SOPP(opcode) => self.parse_sopp_inst(op, lo, opcode),
            Opcode::SMEM(opcode) => {
                let hi = self.read()?;
                self.parse_smem_inst(op, lo, hi, opcode)
            }
            Opcode::VOP1(opcode) => self.parse_vop1_inst(op, lo, opcode),
            Opcode::VOP2(opcode) => self.parse_vop2_inst(op, lo, opcode),
            Opcode::VOPC(opcode) => self.parse_vopc_inst(op, lo, opcode),
            Opcode::VOP3P(opcode) => {
                let hi = self.read()?;
                self.parse_vop3p_inst(op, lo, hi, opcode)
            }
            // Opcode::VINTERP => {}
            Opcode::LDSGDS(opcode) => {
                let hi = self.read()?;
                self.parse_ldsgds_inst(op, lo, hi, opcode)
            }
            Opcode::VOP3AB(opcode) => {
                let hi = self.read()?;
                self.parse_vop3ab_inst(op, lo, hi, opcode)
            }
            Opcode::MUBUF(opcode) => {
                let hi = self.read()?;
                self.parse_mubuf_inst(op, lo, hi, opcode)
            }
            // Opcode::MTBUF => {}
            // Opcode::MIMG => {}
            // Opcode::EXPORT => {}
            Opcode::VMEM(opcode) => {
                let hi = self.read()?;
                self.parse_vmem_inst(op, lo, hi, opcode)
            }

            Opcode::INVALID(code) => Ok(Instruction::invalid(code)),
            _ => todo!(),
        }
    }

    fn parse_sop1_inst(&mut self, op: Opcode, lo: u32, opcode: SOP1Opcode) -> Result<Instruction> {
        use DecodeOperand::*;
        use SOP1InstExpectedOperands::*;
        use SOP1InstSideEffect::*;
        let expected_operands = SOP1InstExpectedOperands::from_opcode(opcode);
        let side_effect = SOP1InstSideEffect::from_opcode(opcode);
        let (dst_size, src_size) = (opcode.dst_size(), opcode.src_size());
        match (side_effect, expected_operands) {
            (NoSideEffect, AccessDstSrc) => {
                let desc = &[
                    ScalarDst(16, 23, dst_size as u8),
                    ScalarSrc(0, 8, src_size as u8),
                ];
                self.parse_alu_inst(op, lo, desc)
            }
            (NoSideEffect, NoSrc) => {
                let desc = &[ScalarDst(16, 23, dst_size as u8)];
                self.parse_alu_inst(op, lo, desc)
            }
            (NoSideEffect, NoDst) => {
                let desc = &[Void, ScalarSrc(0, 8, src_size as u8)];
                self.parse_alu_inst(op, lo, desc)
            }
            (ReadScc, AccessDstSrc) | (WriteScc, AccessDstSrc) => {
                let desc = &[
                    ScalarDst(16, 23, dst_size as u8),
                    SpecialScalarReg(Operand::SPECIAL_REG_SCC),
                    ScalarSrc(0, 8, src_size as u8),
                ];
                self.parse_alu_inst(op, lo, desc)
            }
            (ReadM0, AccessDstSrc) => {
                let desc = &[
                    ScalarDst(16, 23, dst_size as u8),
                    SpecialScalarReg(Operand::SPECIAL_REG_M0),
                    ScalarSrc(0, 8, src_size as u8),
                ];
                self.parse_alu_inst(op, lo, desc)
            }
            (WriteSccAndExec, AccessDstSrc) => {
                let desc = &[
                    ScalarDst(16, 23, dst_size as u8),
                    SpecialScalarReg(Operand::SPECIAL_REG_EXEC_LO),
                    SpecialScalarReg(Operand::SPECIAL_REG_SCC),
                    ScalarSrc(0, 8, src_size as u8),
                ];
                self.parse_alu_inst(op, lo, desc)
            }
            _ => {
                // no such case in ISA
                Err(Error::DecodeError(DecodeError::InvalidInstruction))?
            }
        }
    }

    fn parse_sopc_inst(&mut self, op: Opcode, lo: u32, opcode: SOPCOpcode) -> Result<Instruction> {
        use DecodeOperand::*;
        let (inst_type, data_size) = opcode.get_type_size();
        let (w0, w1) = inst_type.src_size(data_size);
        let desc = &[
            SpecialScalarReg(Operand::SPECIAL_REG_SCC),
            ScalarSrc(0, 8, w0 as u8),
            ScalarSrc(8, 16, w1 as u8),
        ];
        self.parse_alu_inst(op, lo, desc)
    }

    fn parse_sop2_inst(&mut self, op: Opcode, lo: u32, opcode: SOP2Opcode) -> Result<Instruction> {
        use DecodeOperand::*;
        use SOP2InstSideEffect::*;
        let (dst_size, src0_size, src1_size) = opcode.reg_size();
        match SOP2InstSideEffect::from_opcode(opcode) {
            NoSideEffect => {
                let desc = &[
                    ScalarDst(16, 23, dst_size as u8),
                    ScalarSrc(0, 8, src0_size as u8),
                    ScalarSrc(8, 16, src1_size as u8),
                ];
                self.parse_alu_inst(op, lo, desc)
            }
            ReadScc | WriteScc | ReadWriteScc => {
                let desc = &[
                    ScalarDst(16, 23, dst_size as u8),
                    SpecialScalarReg(Operand::SPECIAL_REG_SCC),
                    ScalarSrc(0, 8, src0_size as u8),
                    ScalarSrc(8, 16, src1_size as u8),
                ];
                self.parse_alu_inst(op, lo, desc)
            }
        }
    }

    fn parse_sopk_inst(&mut self, op: Opcode, lo: u32, opcode: SOPKOpcode) -> Result<Instruction> {
        use DecodeOperand::*;
        use SOPKOpcode::*;
        match opcode {
            S_VERSION => {
                const DESC: [DecodeOperand; 2] = [Void, ScalarImm(0, 16)];
                self.parse_alu_inst(op, lo, &DESC)
            }
            S_CMOVK_I32 => {
                const DESC: [DecodeOperand; 3] = [
                    ScalarSrc(16, 23, 1),
                    ScalarSignedImm(0, 16),
                    SpecialScalarReg(Operand::SPECIAL_REG_SCC),
                ];
                self.parse_alu_inst(op, lo, &DESC)
            }
            S_WAITCNT_EXPCNT | S_WAITCNT_LGKMCNT | S_WAITCNT_VMCNT | S_WAITCNT_VSCNT => {
                const DESC: [DecodeOperand; 3] = [Void, ScalarSrc(16, 23, 1), ScalarImm(0, 16)];
                self.parse_alu_inst(op, lo, &DESC)
            }
            S_MOVK_I32 | S_MULK_I32 => {
                const DESC: [DecodeOperand; 2] = [ScalarDst(16, 23, 1), ScalarSignedImm(0, 16)];
                self.parse_alu_inst(op, lo, &DESC)
            }
            S_SUBVECTOR_LOOP_BEGIN | S_SUBVECTOR_LOOP_END | S_GETREG_B32 => {
                const DESC: [DecodeOperand; 2] = [ScalarDst(16, 23, 1), ScalarImm(0, 16)];
                self.parse_alu_inst(op, lo, &DESC)
            }
            S_CALL_B64 => {
                const DESC: [DecodeOperand; 2] = [ScalarDst(16, 23, 2), ScalarImm(0, 16)];
                self.parse_alu_inst(op, lo, &DESC)
            }
            S_ADDK_I32 => {
                const DESC: [DecodeOperand; 3] = [
                    ScalarDst(16, 23, 1),
                    SpecialScalarReg(Operand::SPECIAL_REG_SCC),
                    ScalarSignedImm(0, 16),
                ];
                self.parse_alu_inst(op, lo, &DESC)
            }
            S_SETREG_IMM32_B32 => {
                const DESC: [DecodeOperand; 3] = [Void, ScalarImm(0, 16), Literal];
                self.parse_alu_inst(op, lo, &DESC)
            }
            S_SETREG_B32 => {
                const DESC: [DecodeOperand; 3] = [Void, ScalarImm(0, 16), ScalarSrc(16, 23, 1)];
                self.parse_alu_inst(op, lo, &DESC)
            }
            S_CMPK_EQ_I32 | S_CMPK_GE_I32 | S_CMPK_GT_I32 | S_CMPK_LE_I32 | S_CMPK_LG_I32
            | S_CMPK_LT_I32 => {
                const DESC: [DecodeOperand; 3] = [
                    SpecialScalarReg(Operand::SPECIAL_REG_SCC),
                    ScalarSrc(16, 23, 1),
                    ScalarSignedImm(0, 16),
                ];
                self.parse_alu_inst(op, lo, &DESC)
            }
            _ => {
                // S_CMPK_*_U32
                const DESC: [DecodeOperand; 3] = [
                    SpecialScalarReg(Operand::SPECIAL_REG_SCC),
                    ScalarSrc(16, 23, 1),
                    ScalarImm(0, 16),
                ];
                self.parse_alu_inst(op, lo, &DESC)
            }
        }
    }

    fn parse_sopp_inst(&mut self, op: Opcode, lo: u32, opcode: SOPPOpcode) -> Result<Instruction> {
        use DecodeOperand::*;
        use SOPPInstExpectedOperands::*;
        match SOPPInstExpectedOperands::from_opcode(opcode) {
            ReadSimm => self.parse_alu_inst(op, lo, &[Void, ScalarSignedImm(0, 16)]),
            NoOperand => self.parse_alu_inst(op, lo, &[Void]),
            ConditionalBranch(operand) => {
                if let Operand::SpecialScalarRegister(idx) = operand {
                    self.parse_alu_inst(
                        op,
                        lo,
                        &[Void, SpecialScalarReg(idx), ScalarSignedImm(0, 16)],
                    )
                } else {
                    unreachable!("Invalid special register for conditional branches")
                }
            }
        }
    }

    fn parse_smem_inst(
        &mut self,
        op: Opcode,
        lo: u32,
        hi: u32,
        opcode: SMEMOpcode,
    ) -> Result<Instruction> {
        use DecodeOperand::*;
        use SMEMInstExpectedOperands::*;
        let data_size = opcode.data_size();
        let addr_size = opcode.addr_size();
        let operands = match opcode.expected_operands() {
            NoOperands => {
                smallvec![Operand::Void]
            }
            NoSrc => {
                let desc = [ScalarDst(6, 13, data_size as u8)];
                self.parse_operands(lo, &desc)?
            }
            NoDst => {
                let desc_lo = [
                    Void,
                    ScalarImm(6, 13),
                    ScalarRegSlice(0, 6, addr_size as u8, 1),
                ];
                let desc_hi = [ScalarSrc(25, 32, 1), ScalarSignedImm(0, 21)];
                let mut operands = self.parse_operands(lo, &desc_lo)?;
                let op_hi = self.parse_operands(hi, &desc_hi)?;
                operands.extend(op_hi);
                operands
            }
            AccessDstSrc => {
                let desc_lo = [
                    ScalarDst(6, 13, data_size as u8),
                    ScalarRegSlice(0, 6, addr_size as u8, 1),
                    ScalarImm(14, 15),
                    ScalarImm(16, 17),
                ];
                let desc_hi = [ScalarSrc(25, 32, 1), ScalarSignedImm(0, 21)];
                let mut operands = self.parse_operands(lo, &desc_lo)?;
                let op_hi = self.parse_operands(hi, &desc_hi)?;
                operands.extend(op_hi);
                operands
            }
        };
        Ok(Instruction {
            op,
            operands,
            modifier: None,
        })
    }

    fn parse_ldsgds_inst(
        &mut self,
        op: Opcode,
        lo: u32,
        hi: u32,
        opcode: LDSGDSOpcode,
    ) -> Result<Instruction> {
        use DSInstSideEffect::*;
        use DecodeOperand::*;
        let (dst_count, addr_count, data_count) = opcode.expected_operands();
        let reg_size = opcode.reg_size() as u8;
        const ADDR: DecodeOperand = VectorReg(0, 8, 1);
        let data0 = VectorReg(8, 16, reg_size);
        let data1 = VectorReg(16, 24, reg_size);
        let vdst = VectorReg(24, 32, reg_size * dst_count);
        const M0: DecodeOperand = SpecialScalarReg(Operand::SPECIAL_REG_M0);
        let side_effect = opcode.side_effect();
        let desc_hi: SmallVec<[DecodeOperand; 4]> =
            match (dst_count, addr_count, data_count, side_effect) {
                (0, 0, 0, NoSideEffect) => smallvec![Void],
                (0, 0, 0, ReadM0) => smallvec![Void, M0],
                (0, 0, 1, ReadM0) => smallvec![Void, M0, data0],
                (0, 1, 0, ReadM0) => smallvec![Void, M0, ADDR],
                (0, 1, 1, NoSideEffect) => smallvec![Void, ADDR, data0],
                (0, 1, 2, NoSideEffect) => smallvec![Void, ADDR, data0, data1],
                (_, 0, 0, ReadM0) => smallvec![vdst, M0],
                (_, 1, 0, NoSideEffect) => smallvec![vdst, ADDR],
                (_, 1, 1, NoSideEffect) => smallvec![vdst, ADDR, data0],
                (_, 1, 2, NoSideEffect) => smallvec![vdst, ADDR, data0, data1],
                _ => Err(Error::DecodeError(DecodeError::InvalidInstruction))?,
            };
        Ok(Instruction {
            op,
            operands: self.parse_operands(hi, &desc_hi)?,
            modifier: Some(InstructionModifier::Ds(DSModifier(lo))),
        })
    }

    fn parse_vop3ab_inst(
        &mut self,
        op: Opcode,
        lo: u32,
        hi: u32,
        opcode: VOP3ABOpcode,
    ) -> Result<Instruction> {
        use DecodeOperand::*;
        use VOP3ABDstType::*;
        let (dst_size, src0_size, src1_size, src2_size) = opcode.reg_size();
        let modifier = if opcode.is_vop3b() {
            VOP3Modifier::from_vop3b_inst(lo, hi)
        } else {
            VOP3Modifier::from_vop3a_inst(lo, hi)
        };
        let desc_lo: SmallVec<[_; 2]> = match VOP3ABDstType::from_opcode(opcode) {
            VDst => smallvec![VectorReg(0, 8, dst_size)],
            VDstSDst => smallvec![VectorReg(0, 8, dst_size), ScalarDst(8, 15, 1)],
            VDstAsSGPR => smallvec![ScalarDst(0, 8, dst_size)],
            Exec => smallvec![SpecialScalarReg(Operand::SPECIAL_REG_EXEC_LO)],
            NoDst => smallvec![Void],
        };
        let mut operands = self.parse_operands(lo, &desc_lo)?;
        let desc_hi = [
            VectorSrc(0, 9, src0_size),
            VectorSrc(9, 18, src1_size),
            VectorSrc(18, 27, src2_size),
        ];
        let op_hi = self.parse_operands(hi, &desc_hi[0..opcode.src_count()])?;
        operands.extend(op_hi);
        let modifier = Some(InstructionModifier::Vop3(modifier));
        Ok(Instruction {
            op,
            operands,
            modifier,
        })
    }

    fn parse_vop3p_inst(
        &mut self,
        op: Opcode,
        lo: u32,
        hi: u32,
        opcode: VOP3POpcode,
    ) -> Result<Instruction> {
        use DecodeOperand::*;
        let modifier = VOP3PModifier::from_inst(lo, hi);
        const DESC_LO: [DecodeOperand; 1] = [VectorReg(0, 8, 1)];
        const DESC_HI: [DecodeOperand; 3] = [
            VectorSrc(0, 9, 1),
            VectorSrc(9, 18, 1),
            VectorSrc(18, 27, 1),
        ];
        let mut operands = self.parse_operands(lo, &DESC_LO)?;
        let op_hi = self.parse_operands(hi, &DESC_HI[0..opcode.src_count()])?;
        operands.extend(op_hi);
        let modifier = Some(InstructionModifier::Vop3p(modifier));
        Ok(Instruction {
            op,
            operands,
            modifier,
        })
    }

    fn parse_vmem_inst(
        &mut self,
        op: Opcode,
        lo: u32,
        hi: u32,
        opcode: VMEMOpcode,
    ) -> Result<Instruction> {
        use DecodeOperand::*;
        let modifier = VMEMModifier::from_instruction(lo, hi);
        let (vdst_size, vdata_size) = opcode.data_size();
        // offset, flags
        const DESC_LO: [DecodeOperand; 1] = [ScalarSignedImm(0, 12)];
        // vdst, addr, data, saddr
        let desc_hi = [
            VectorReg(24, 32, vdst_size as u8),
            VectorReg(0, 8, modifier.vaddr_size() as u8),
            VectorReg(8, 16, vdata_size as u8),
            ScalarSrc(16, 23, 2),
        ];
        let op_lo = self.parse_operands(lo, &DESC_LO)?;
        let mut operands = self.parse_operands(hi, &desc_hi)?;
        operands.extend(op_lo);
        Ok(Instruction {
            op,
            operands,
            modifier: Some(InstructionModifier::VMem(modifier)),
        })
    }

    fn parse_mubuf_inst(
        &mut self,
        op: Opcode,
        lo: u32,
        hi: u32,
        opcode: MUBUFOpcode,
    ) -> Result<Instruction> {
        use DecodeOperand::*;
        use MUBUFInstructionType::*;
        let modifier = MUBUFModifier::from_instrcution(lo, hi);
        let vdata_len = opcode.reg_len() as u8;
        let vaddr_len = modifier.vaddr_size();
        let operands = match MUBUFInstructionType::from_opcode(opcode) {
            BufferAtomic | BufferLoad => {
                // vdata, vaddr, srsrc, soffset, slc
                let desc_hi = [
                    VectorReg(8, 16, vdata_len),
                    VectorReg(0, 8, vaddr_len),
                    ScalarRegSlice(16, 21, 4, 2),
                    ScalarSrc(24, 32, 1),
                ];
                self.parse_operands(hi, &desc_hi)?
            }
            BufferStore => {
                // void, vdata, vaddr, srsrc, soffset, slc
                let desc_hi = [
                    Void,
                    VectorReg(8, 16, vdata_len),
                    VectorReg(0, 8, vaddr_len),
                    ScalarRegSlice(16, 21, 4, 2),
                    ScalarSrc(24, 32, 1),
                ];
                self.parse_operands(hi, &desc_hi)?
            }
            BufferGL => smallvec![Operand::Void],
        };
        Ok(Instruction {
            op,
            operands,
            modifier: Some(InstructionModifier::Mubuf(modifier)),
        })
    }

    fn parse_vop1_inst(&mut self, op: Opcode, lo: u32, opcode: VOP1Opcode) -> Result<Instruction> {
        use DecodeOperand::*;
        use VOP1Opcode::*;
        let modifier = self.parse_modifier(lo, 0, 9)?;
        let (dst_size, src_size) = opcode.reg_size();
        let src0 = match modifier {
            Some(InstructionModifier::Sdwa(sdwa)) => SDWASrc0(sdwa),
            Some(InstructionModifier::Dpp16(dpp)) => Dpp16Src0(dpp),
            None => VectorSrc(0, 9, src_size as u8),
            _ => Err(Error::DecodeError(DecodeError::InvalidModifier))?,
        };
        let desc: SmallVec<[_; 4]> = match opcode {
            V_CLREXCP | V_PIPEFLUSH | V_NOP => smallvec![Void],
            V_READFIRSTLANE_B32 => smallvec![ScalarDst(17, 25, 1), src0],
            V_MOVRELD_B32 | V_MOVRELS_B32 | V_MOVRELSD_B32 | V_MOVRELSD_2_B32 | V_SWAPREL_B32 => {
                smallvec![
                    VectorReg(17, 25, dst_size as u8),
                    SpecialScalarReg(Operand::SPECIAL_REG_M0),
                    src0,
                ]
            }
            _ => smallvec![VectorReg(17, 25, dst_size as u8), src0],
        };
        self.parse_inst_with_modifier(op, lo, &desc, modifier)
    }

    fn parse_vopc_inst(&mut self, op: Opcode, lo: u32, opcode: VOPCOpcode) -> Result<Instruction> {
        use DecodeOperand::*;
        let (inst_type, data_size) = opcode
            .get_type_size()
            .ok_or(Error::DecodeError(DecodeError::InvalidOperand))?;
        let (s0_size, s1_size) = inst_type.src_size(data_size);
        let default_dst = if VOPCOpcodeFlags::from_opcode(opcode).write_exec() {
            Operand::SPECIAL_REG_EXEC_LO
        } else {
            Operand::SPECIAL_REG_VCC_LO
        };
        let modifier = self.parse_modifier(lo, 0, 9)?;
        let desc: SmallVec<[_; 4]> = match modifier {
            Some(InstructionModifier::Sdwa(sdwa)) => smallvec![
                SDWABDst(sdwa, default_dst),
                SDWASrc0(sdwa),
                SDWASrc1(sdwa, 9, 17),
            ],
            None => smallvec![
                SpecialScalarReg(default_dst),
                VectorSrc(0, 9, s0_size as u8),
                VectorReg(9, 17, s1_size as u8),
            ],
            _ => Err(Error::DecodeError(DecodeError::InvalidModifier))?,
        };
        self.parse_inst_with_modifier(op, lo, &desc, modifier)
    }

    fn parse_vop2_inst(&mut self, op: Opcode, lo: u32, opcode: VOP2Opcode) -> Result<Instruction> {
        use DecodeOperand::*;
        use VOP2InstExpectedOperands::*;
        use VOP2InstSideEffect::*;
        const VCC: DecodeOperand = SpecialScalarReg(Operand::SPECIAL_REG_VCC_LO);
        let side_effect = VOP2InstSideEffect::from_opcode(opcode);
        let expected_operands = VOP2InstExpectedOperands::from_opcode(opcode);
        let modifier = self.parse_modifier(lo, 0, 9)?;
        let (src0, src1) = match modifier {
            Some(InstructionModifier::Sdwa(sdwa)) => (SDWASrc0(sdwa), SDWASrc1(sdwa, 9, 17)),
            Some(InstructionModifier::Dpp16(dpp)) => (Dpp16Src0(dpp), VectorReg(9, 17, 1)),
            None => (VectorSrc(0, 9, 1), VectorReg(9, 17, 1)),
            _ => Err(Error::DecodeError(DecodeError::InvalidModifier))?,
        };
        let desc: SmallVec<[_; 4]> = match (side_effect, expected_operands) {
            (NoSideEffect, TwoSrcs) => {
                smallvec![VectorReg(17, 25, 1), src0, src1]
            }
            (NoSideEffect, ExtraSimmAsSrc1) => smallvec![VectorReg(17, 25, 1), src0, Literal, src1],
            (NoSideEffect, ExtraSimmAsSrc2) => smallvec![VectorReg(17, 25, 1), src0, src1, Literal],
            (ReadVcc, TwoSrcs) => smallvec![VectorReg(17, 25, 1), src0, src1, VCC],
            (ReadAndWriteVcc, TwoSrcs) => smallvec![VectorReg(17, 25, 1), VCC, src0, src1, VCC],
            _ => {
                // no such case in ISA
                Err(Error::DecodeError(DecodeError::InvalidInstruction))?
            }
        };
        self.parse_inst_with_modifier(op, lo, &desc, modifier)
    }

    fn parse_modifier(&mut self, v: u32, lo: u32, hi: u32) -> Result<Option<InstructionModifier>> {
        match Self::extract(v, lo, hi) {
            Operand::SDWA => {
                let v = self.get_or_read_literal()?;
                Ok(Some(InstructionModifier::Sdwa(SDWAModifier(v))))
            }
            Operand::DPP16 => {
                let v = self.get_or_read_literal()?;
                Ok(Some(InstructionModifier::Dpp16(DPP16Modifier(v))))
            }
            _ => Ok(None),
        }
    }

    fn parse_operands(
        &mut self,
        v: u32,
        descriptors: &[DecodeOperand],
    ) -> Result<SmallVec<[Operand; 4]>> {
        let mut operands = SmallVec::new();
        descriptors.iter().try_for_each(|o| {
            let r = match o {
                DecodeOperand::ScalarDst(lo, hi, len) => self.parse_operand(
                    Self::extract(v, (*lo).into(), (*hi).into()),
                    DecodeOption::new().reg_len(*len),
                ),
                DecodeOperand::ScalarSrc(lo, hi, len) => self.parse_operand(
                    Self::extract(v, (*lo).into(), (*hi).into()),
                    DecodeOption::new().constant().literal().reg_len(*len),
                ),
                DecodeOperand::ScalarRegSlice(lo, hi, len, missed_count) => self.parse_operand(
                    Self::extract(v, (*lo).into(), (*hi).into()) << missed_count,
                    DecodeOption::new().scalar_reg_slice(*len),
                ),
                DecodeOperand::ScalarImm(lo, hi) => {
                    Ok(Operand::Constant(Self::extract(v, *lo, *hi) as i32))
                }
                DecodeOperand::ScalarSignedImm(lo, hi) => {
                    let v = Self::extract(v, *lo, *hi);
                    // Only support a few fixed lengths for signed immediates
                    let len = hi - lo;
                    match len {
                        16 => Ok(Operand::Constant(v as i16 as i32)),
                        8 => Ok(Operand::Constant(v as i8 as i32)),
                        12 | 21 => {
                            let v = if (v >> (len - 1)) == 1 {
                                (v as i32) - (1 << len)
                            } else {
                                v as i32
                            };
                            Ok(Operand::Constant(v))
                        }
                        _ => todo!("Unsupported length for signed immediate"),
                    }
                }
                DecodeOperand::Void => Ok(Operand::Void),
                DecodeOperand::SpecialScalarReg(r) => Ok(Operand::SpecialScalarRegister(*r)),
                DecodeOperand::VectorReg(lo, hi, len) => Ok(Operand::VectorRegister(
                    Self::extract(v, *lo as u32, *hi as u32) as u8,
                    *len,
                )),
                DecodeOperand::VectorSrc(lo, hi, len) => self.parse_operand(
                    Self::extract(v, (*lo).into(), (*hi).into()),
                    DecodeOption::new()
                        .constant()
                        .literal()
                        .vreg()
                        .reg_len(*len),
                ),
                DecodeOperand::Literal => Ok(Operand::Constant(self.get_or_read_literal()? as i32)),
                DecodeOperand::SDWABDst(modifier, default) => modifier.parse_sdwab_dst(*default),
                DecodeOperand::SDWASrc0(modifier) => modifier.parse_src0(),
                DecodeOperand::SDWASrc1(modifier, lo, hi) => {
                    modifier.parse_src1(Self::extract(v, (*lo).into(), (*hi).into()) as u8)
                }
                DecodeOperand::Dpp16Src0(modifier) => Ok(modifier.parse_src0()),
            }?;
            operands.push(r);
            Ok::<(), Error>(())
        })?;
        Ok(operands)
    }

    fn parse_inst_with_modifier(
        &mut self,
        op: Opcode,
        v: u32,
        descriptors: &[DecodeOperand],
        modifier: Option<InstructionModifier>,
    ) -> Result<Instruction> {
        let operands = self.parse_operands(v, descriptors)?;
        Ok(Instruction {
            op,
            operands,
            modifier,
        })
    }

    fn parse_alu_inst(
        &mut self,
        op: Opcode,
        v: u32,
        descriptors: &[DecodeOperand],
    ) -> Result<Instruction> {
        self.parse_inst_with_modifier(op, v, descriptors, None)
    }

    #[inline]
    fn extract(v: u32, lo: u32, hi: u32) -> u32 {
        (v >> lo) & ((1u32 << (hi - lo)) - 1)
    }

    fn parse_operand(&mut self, v: u32, option: DecodeOption) -> Result<Operand> {
        use Operand::*;
        let disallow_constant = !option.contains(DecodeOption::CONSTANT);
        let disallow_literal = !option.contains(DecodeOption::LITERAL);
        let disallow_vreg = !option.contains(DecodeOption::VREG);
        let disallow_special = option.contains(DecodeOption::SCALAR_REG_SLICE);
        if v <= 105 {
            Ok(ScalarRegister(v as u8, option.get_reg_len()))
        } else if (106..=127).contains(&v) || (251..=254).contains(&v) {
            if disallow_special {
                Err(Error::DecodeError(DecodeError::InvalidOperand))
            } else {
                Ok(SpecialScalarRegister(v as u8))
            }
        } else if (128..=208).contains(&v) {
            if disallow_constant {
                Err(Error::DecodeError(DecodeError::InvalidOperand))
            } else if v <= 192 {
                Ok(Constant((v - 128) as i32))
            } else {
                Ok(Constant(192 - (v as i32)))
            }
        } else if (240..=248).contains(&v) {
            if disallow_constant {
                Err(Error::DecodeError(DecodeError::InvalidOperand))
            } else {
                Ok(ConstantFloat((v - 240) as u8))
            }
        } else if v == 255 {
            if disallow_constant || disallow_literal {
                Err(Error::DecodeError(DecodeError::InvalidOperand))
            } else {
                Ok(Constant(self.get_or_read_literal()? as i32))
            }
        } else if (256..512).contains(&v) {
            if disallow_vreg {
                Err(Error::DecodeError(DecodeError::InvalidOperand))
            } else {
                Ok(VectorRegister((v - 256) as u8, option.get_reg_len()))
            }
        } else {
            Err(Error::DecodeError(DecodeError::InvalidOperand))
        }
    }
}

impl<'a> Iterator for Decoder<'a> {
    type Item = (usize, Instruction);

    /// Return value:
    /// - offset and the next instruction if success
    /// - offset and `INVALID` if the next dword cannot be decoded correctly
    /// - None if other unhandled exceptions (e.g. unexpected EOF) occur
    fn next(&mut self) -> Option<Self::Item> {
        let off = self.offset();
        match self.decode_some() {
            Ok(x) => Some((off, x)),
            Err(x) => match x {
                Error::DecodeError(_) => {
                    self.offset = off; // reset offset and read again
                    if let Ok(code) = self.read() {
                        Some((off, Instruction::invalid(code)))
                    } else {
                        None
                    }
                }
                _ => None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse() {
        const BINARY: [u32; 25] = [
            0xf4000082, 0xfa000004, 0xf4040003, 0xfa000008, 0xbf8cc07f, 0x8702ff02, 0x0000ffff,
            0x93080208, 0x4a000008, 0x7d880000, 0xbe803c6a, 0xbf88000c, 0xf4040083, 0xfa000000,
            0x7e020280, 0x7e040201, 0xd6ff0000, 0x00020082, 0xbf8cc07f, 0xd70f6a00, 0x00020002,
            0x50020203, 0xdc708000, 0x007d0200, 0xbf810000,
        ];
        let stream = Decoder::new(&BINARY);
        let instructions: Vec<(usize, Instruction)> = stream.collect();
        assert_eq!(18, instructions.len());
        assert!(!instructions
            .iter()
            .any(|(_, x)| matches!(x.op, Opcode::INVALID(_))));
    }

    fn test_instructions(binaries: &[&[u32]], asm_codes: &[&str]) {
        binaries.iter().zip(asm_codes).for_each(|(binary, asm)| {
            let inst = Decoder::new(&binary)
                .decode_some()
                .expect("Can not decode instruction");
            assert_eq!(&format!("{}", inst), asm);
        });
    }

    #[test]
    fn test_parse_operand() {
        use Operand::*;
        // the next dword for literal operand
        const BINARY: [u32; 1] = [0x12345678];
        let parse = |code, options| {
            Decoder::new(&BINARY)
                .parse_operand(code, options)
                .expect("Can not decode operand")
        };
        assert!(matches!(
            parse(0x12, DecodeOption::new()),
            ScalarRegister(0x12, 1)
        ));
        assert!(matches!(
            parse(Operand::SPECIAL_REG_M0 as u32, DecodeOption::new()),
            SpecialScalarRegister(Operand::SPECIAL_REG_M0)
        ));
        assert!(matches!(
            parse(0xab, DecodeOption::new().constant()),
            Constant(0x2b)
        ));
        assert!(matches!(
            parse(0xff, DecodeOption::new().constant().literal()),
            Constant(0x12345678)
        ));
        assert!(matches!(
            parse(0x123, DecodeOption::new().vreg()),
            VectorRegister(0x23, 1)
        ));
        let parse_fail = |code, options| {
            Decoder::new(&BINARY)
                .parse_operand(code, options)
                .expect_err("Should not decode this operand");
        };
        parse_fail(0xab, DecodeOption::new());
        parse_fail(
            Operand::SPECIAL_REG_M0 as u32,
            DecodeOption::new().scalar_reg_slice(1),
        );
        parse_fail(0xff, DecodeOption::new().constant());
        parse_fail(0x123, DecodeOption::new());
    }

    #[test]
    fn test_multi_literals() {
        // It is legitimate for both operands to point to the same literal
        const BINARY: [&[u32]; 1] = [&[0x8100ffff, 0x80000000]];
        const ASM: [&str; 1] = ["s_add_i32 s0, 0x80000000, 0x80000000"];
        test_instructions(&BINARY, &ASM);
    }

    #[test]
    fn test_fmt_neg_hex() {
        const IMM_STR: [(i32, &'static str); 5] = [
            (0x0, "0x0"),
            (0x1234, "0x1234"),
            (0x65536, "0x65536"),
            (-0x1234, "-0x1234"),
            (-0x65536, "-0x65536"),
        ];
        IMM_STR
            .iter()
            .for_each(|(v, s)| assert_eq!(&format!("{:#x}", Operand::Constant(*v)), s));
    }

    #[test]
    fn test_decode_signed_imm() {
        use DecodeOperand::ScalarSignedImm;
        const OP_VALUE: [(u32, i32, u32); 4] = [
            (0x0, 0, 21),
            (0x1fedcc, -0x1234, 21),
            (0x1234, 0x1234, 21),
            (0xedd, -0x123, 12),
        ];
        OP_VALUE.iter().for_each(|(x, v, len)| {
            let operands = Decoder::new(&[])
                .parse_operands(*x, &[ScalarSignedImm(0, *len)])
                .expect("Can not decode operands");
            match operands.as_slice() {
                [Operand::Constant(x)] => assert_eq!(x, v),
                _ => panic!("Incorrect decoding result"),
            }
        });
    }

    #[test]
    fn test_special_sreg() {
        const BINARY: [&[u32]; 8] = [
            &[0xBE80036A],
            &[0xBE80037D],
            &[0xBE80037C],
            &[0xBE80037E],
            &[0xBE8003FB],
            &[0xBE8003FC],
            &[0xBE8003FD],
            &[0xD4C2006A, 0x000000FE],
        ];
        const ASM: [&str; 8] = [
            "s_mov_b32 s0, vcc_lo",
            "s_mov_b32 s0, null",
            "s_mov_b32 s0, m0",
            "s_mov_b32 s0, exec_lo",
            "s_mov_b32 s0, src_vccz",
            "s_mov_b32 s0, src_execz",
            "s_mov_b32 s0, src_scc",
            "v_cmp_eq_u32_e64 vcc_lo, src_lds_direct, s0",
        ];
        test_instructions(&BINARY, &ASM);
    }

    #[test]
    fn test_sopk() {
        const BINARY: [&[u32]; 16] = [
            &[0xB0802004],
            &[0xB0000800],
            &[0xBBFD0000],
            &[0xB5010000],
            &[0xBB260B7A],
            &[0xB4A50000],
            &[0xB5BC0000],
            &[0xBD7D0000],
            &[0xB7A00200],
            &[0xB100FFFF],
            &[0xB000FFFF],
            &[0xB780FFFF],
            &[0xB800FFFF],
            &[0xB200FFFF],
            &[0xB180FFFF],
            &[0xB280FFFF],
        ];
        const ASM: [&str; 16] = [
            "s_version 0x2004",
            "s_movk_i32 s0, 0x800",
            "s_waitcnt_vscnt null, 0x0",
            "s_cmpk_lg_u32 s1, 0x0",
            "s_call_b64 s[38:39], 2938",
            "s_cmpk_eq_u32 s37, 0x0",
            "s_cmpk_gt_u32 s60, 0x0",
            "s_waitcnt_lgkmcnt null, 0x0",
            "s_addk_i32 s32, 0x200",
            "s_cmovk_i32 s0, 0xffff",
            "s_movk_i32 s0, 0xffff",
            "s_addk_i32 s0, 0xffff",
            "s_mulk_i32 s0, 0xffff",
            "s_cmpk_lg_i32 s0, 0xffff",
            "s_cmpk_eq_i32 s0, 0xffff",
            "s_cmpk_gt_i32 s0, 0xffff",
        ];
        test_instructions(&BINARY, &ASM);
        // test for s_addk_i32
        let inst = Decoder::new(&[0xB7A00200])
            .decode_some()
            .expect("Can not decode instruction");
        assert!(matches!(
            inst.operands[1],
            Operand::SpecialScalarRegister(Operand::SPECIAL_REG_SCC)
        ));
    }

    #[test]
    fn test_hwreg() {
        const BINARY: [&[u32]; 4] = [
            &[0xB988F814],
            &[0xBA80F814, 0x0000FFFF],
            &[0xB908F814],
            &[0xB9881881],
        ];
        const ASM: [&str; 4] = [
            "s_setreg_b32 hwreg(HW_REG_FLAT_SCR_LO), s8",
            "s_setreg_imm32_b32 hwreg(HW_REG_FLAT_SCR_LO), 0xffff",
            "s_getreg_b32 s8, hwreg(HW_REG_FLAT_SCR_LO)",
            "s_setreg_b32 hwreg(HW_REG_MODE, 2, 4), s8",
        ];
        test_instructions(&BINARY, &ASM);
    }

    #[test]
    fn test_vmem() {
        const BINARY: [&[u32]; 24] = [
            &[0xDC208000, 0x050C0006],
            &[0xDC288000, 0x067D0006],
            &[0xDC308000, 0x01000001],
            &[0xDC359018, 0x047D0000],
            &[0xDC388000, 0x007D0000],
            &[0xDC3C8004, 0x127D0009],
            &[0xDC948000, 0x067D000A],
            &[0xDC608000, 0x000C0706],
            &[0xDC688000, 0x000C2439],
            &[0xDC6C8000, 0x007D0F06],
            &[0xDC708000, 0x007D0004],
            &[0xDC748014, 0x00002229],
            &[0xDC788010, 0x00080038],
            &[0xDC7C8188, 0x00000629],
            &[0xDC339010, 0x01000002],
            &[0xDCC98000, 0x130C1314],
            &[0xDD488008, 0x007D0002],
            &[0xDD458018, 0x067D0200],
            &[0xDC30000C, 0x257D0033],
            &[0xDC340000, 0x237D0033],
            &[0xDC780010, 0x007D1B33],
            &[0xDC340000, 0x007D0033],
            &[0xDC300008, 0x117D0033],
            &[0xDC388FF8, 0x1F7D001D],
        ];
        const ASM: [&str; 24] = [
            "global_load_ubyte v5, v6, s[12:13]",
            "global_load_ushort v6, v[6:7], off",
            "global_load_dword v1, v1, s[0:1]",
            "global_load_dwordx2 v[4:5], v[0:1], off offset:24 glc dlc",
            "global_load_dwordx4 v[0:3], v[0:1], off",
            "global_load_dwordx3 v[18:20], v[9:10], off offset:4",
            "global_load_short_d16_hi v6, v[10:11], off",
            "global_store_byte v6, v7, s[12:13]",
            "global_store_short v57, v36, s[12:13]",
            "global_store_short_d16_hi v[6:7], v15, off",
            "global_store_dword v[4:5], v0, off",
            "global_store_dwordx2 v41, v[34:35], s[0:1] offset:20",
            "global_store_dwordx4 v56, v[0:3], s[8:9] offset:16",
            "global_store_dwordx3 v41, v[6:8], s[0:1] offset:392",
            "global_load_dword v1, v2, s[0:1] offset:16 glc slc dlc",
            "global_atomic_add v19, v20, v19, s[12:13] glc",
            "global_atomic_add_x2 v[2:3], v[0:1], off offset:8",
            "global_atomic_cmpswap_x2 v[6:7], v[0:1], v[2:5], off offset:24 glc",
            "flat_load_dword v37, v[51:52] offset:12",
            "flat_load_dwordx2 v[35:36], v[51:52]",
            "flat_store_dwordx4 v[51:52], v[27:30] offset:16",
            "flat_load_dwordx2 v[0:1], v[51:52]",
            "flat_load_dword v17, v[51:52] offset:8",
            "global_load_dwordx4 v[31:34], v[29:30], off offset:-8",
        ];
        test_instructions(&BINARY, &ASM);
    }

    #[test]
    fn test_sop1() {
        const BINARY: [&[u32]; 13] = [
            &[0xBE8203FF, 0x0000FFFF],
            &[0xBE9804C1],
            &[0xBE8A0538],
            &[0xBE800700],
            &[0xBE930B88],
            &[0xBE921B93],
            &[0xBE921D97],
            &[0xBE841F00],
            &[0xBE80201E],
            &[0xBE9E2104],
            &[0xBE833C00],
            &[0xBE833D00],
            &[0xBE813002],
        ];
        const ASM: [&str; 13] = [
            "s_mov_b32 s2, 0xffff",
            "s_mov_b64 s[24:25], -1",
            "s_cmov_b32 s10, s56", // read scc
            "s_not_b32 s0, s0",    // write scc
            "s_brev_b32 s19, 8",
            "s_bitset0_b32 s18, 19",
            "s_bitset1_b32 s18, 23",
            "s_getpc_b64 s[4:5]",   // no src
            "s_setpc_b64 s[30:31]", // no dst
            "s_swappc_b64 s[30:31], s[4:5]",
            "s_and_saveexec_b32 s3, s0", // write scc & exec
            "s_or_saveexec_b32 s3, s0",  // write scc & exec
            "s_movreld_b32 s1, s2",      // read m0
        ];
        test_instructions(&BINARY, &ASM);
    }

    #[test]
    fn test_sop2() {
        const BINARY: [&'static [u32]; 28] = [
            &[0x8006A006],
            &[0x80840880],
            &[0x81098109],
            &[0x81970380],
            &[0x82078007],
            &[0x82868080],
            &[0x8304A00E],
            &[0x83808F00],
            &[0x850380C1],
            &[0x85EA80C1],
            &[0x87000200],
            &[0x887E137E],
            &[0x88B83438],
            &[0x8900037E],
            &[0x8A7E097E],
            &[0x8F031881],
            &[0x8F9E1881],
            &[0x90099001],
            &[0x90BC9F3C],
            &[0x91019F0F],
            &[0x91C68146],
            &[0x93080008],
            &[0x9383FF02, 0x00080008],
            &[0x951CFF1A, 0x00200000],
            &[0x99081414],
            &[0x9A141414],
            &[0x9ABFFF02, 0x00000080],
            &[0x9B434007],
        ];
        const ASM: [&'static str; 28] = [
            "s_add_u32 s6, s6, 32",
            "s_sub_u32 s4, 0, s8",
            "s_add_i32 s9, s9, 1",
            "s_sub_i32 s23, 0, s3",
            "s_addc_u32 s7, s7, 0",
            "s_subb_u32 s6, 0, 0",
            "s_min_i32 s4, s14, 32",
            "s_min_u32 s0, s0, 15",
            "s_cselect_b32 s3, -1, 0",
            "s_cselect_b64 vcc, -1, 0",
            "s_and_b32 s0, s0, s2",
            "s_or_b32 exec_lo, exec_lo, s19",
            "s_or_b64 s[56:57], s[56:57], s[52:53]",
            "s_xor_b32 s0, exec_lo, s3",
            "s_andn2_b32 exec_lo, exec_lo, s9",
            "s_lshl_b32 s3, 1, s24",
            "s_lshl_b64 s[30:31], 1, s24",
            "s_lshr_b32 s9, s1, 16",
            "s_lshr_b64 s[60:61], s[60:61], 31",
            "s_ashr_i32 s1, s15, 31",
            "s_ashr_i64 s[70:71], s[70:71], 1",
            "s_mul_i32 s8, s8, s0",
            "s_bfe_u32 s3, s2, 0x80008",
            "s_bfe_i64 s[28:29], s[26:27], 0x200000",
            "s_pack_ll_b32_b16 s8, s20, s20",
            "s_pack_hh_b32_b16 s20, s20, s20",
            "s_mul_hi_u32 s63, s2, 0x80",
            "s_mul_hi_i32 s67, s7, s64",
        ];
        test_instructions(&BINARY, &ASM);
    }

    #[test]
    fn test_sopp() {
        const BINARY: [&[u32]; 18] = [
            &[0xBF800000],
            &[0xBF810000],
            &[0xBF820018],
            &[0xBF84FFEA],
            &[0xBF85002B],
            &[0xBF860988],
            &[0xBF87FFF6],
            &[0xBF8805C1],
            &[0xBF89FFDC],
            &[0xBF8A0000],
            &[0xBF8CC07F],
            &[0xBF8C3F21],
            &[0xBF8E0001],
            &[0xBF8F0001],
            &[0xBF9F0000],
            &[0xBFA00002],
            &[0xBFA10001],
            &[0xBF82FA39],
        ];
        const ASM: [&'static str; 18] = [
            "s_nop 0",
            "s_endpgm",
            "s_branch 24",
            "s_cbranch_scc0 65514",
            "s_cbranch_scc1 43",
            "s_cbranch_vccz 2440",
            "s_cbranch_vccnz 65526",
            "s_cbranch_execz 1473",
            "s_cbranch_execnz 65500",
            "s_barrier",
            "s_waitcnt lgkmcnt(0)",
            "s_waitcnt vmcnt(1) expcnt(2)",
            "s_sleep 1",
            "s_setprio 1",
            "s_code_end",
            "s_inst_prefetch 0x2",
            "s_clause 0x1",
            "s_branch 64057",
        ];
        test_instructions(&BINARY, &ASM);
    }

    #[test]
    fn test_sendmsg() {
        const BINARY: [&[u32]; 5] = [
            &[0xBF900001],
            &[0xBF900022],
            &[0xBF900133],
            &[0xBF90004F],
            &[0xBF91000A],
        ];
        const ASM: [&'static str; 5] = [
            "s_sendmsg sendmsg(MSG_INTERRUPT)",
            "s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)",
            "s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_EMIT_CUT, 1)",
            "s_sendmsg sendmsg(MSG_SYSMSG, SYSMSG_OP_TTRACE_PC)",
            "s_sendmsghalt sendmsg(MSG_GET_DOORBELL)",
        ];
        test_instructions(&BINARY, &ASM);
    }

    #[test]
    fn test_sopc() {
        const BINARY: [&'static [u32]; 14] = [
            &[0xBF008105],
            &[0xBF02800F],
            &[0xBF04022A],
            &[0xBF058005],
            &[0xBF06C009],
            &[0xBF078000],
            &[0xBF088200],
            &[0xBF093D02],
            &[0xBF0AA018],
            &[0xBF0B8105],
            &[0xBF0C8003],
            &[0xBF0D8001],
            &[0xBF128000],
            &[0xBF138004],
        ];
        const ASM: [&'static str; 14] = [
            "s_cmp_eq_i32 s5, 1",
            "s_cmp_gt_i32 s15, 0",
            "s_cmp_lt_i32 s42, s2",
            "s_cmp_le_i32 s5, 0",
            "s_cmp_eq_u32 s9, 64",
            "s_cmp_lg_u32 s0, 0",
            "s_cmp_gt_u32 s0, 2",
            "s_cmp_ge_u32 s2, s61",
            "s_cmp_lt_u32 s24, 32",
            "s_cmp_le_u32 s5, 1",
            "s_bitcmp0_b32 s3, 0",
            "s_bitcmp1_b32 s1, 0",
            "s_cmp_eq_u64 s[0:1], 0",
            "s_cmp_lg_u64 s[4:5], 0",
        ];
        test_instructions(&BINARY, &ASM);
    }

    #[test]
    fn test_vopc() {
        const BINARY: [&'static [u32]; 24] = [
            &[0x7C025280],
            &[0x7C084722],
            &[0x7C0E3F1F],
            &[0x7C1A2480],
            &[0x7C444480],
            &[0x7C484412],
            &[0x7D020603],
            &[0x7D06C80D],
            &[0x7D0800FF, 0x00000080],
            &[0x7D0CBD40],
            &[0x7D542680],
            &[0x7D821205],
            &[0x7D840C80],
            &[0x7D860201],
            &[0x7D880A0E],
            &[0x7D8A0880],
            &[0x7D8C0A0E],
            &[0x7DC22690],
            &[0x7DC40D02],
            &[0x7DC6081C],
            &[0x7DCA0680],
            &[0x7DCC081C],
            &[0x7C228480],
            &[0x7DE226FF, 0x000004D2],
        ];
        const ASM: [&'static str; 24] = [
            "v_cmp_lt_f32_e32 vcc_lo, 0, v41",
            "v_cmp_gt_f32_e32 vcc_lo, v34, v35",
            "v_cmp_o_f32_e32 vcc_lo, v31, v31",
            "v_cmp_neq_f32_e32 vcc_lo, 0, v18",
            "v_cmp_eq_f64_e32 vcc_lo, 0, v[34:35]",
            "v_cmp_gt_f64_e32 vcc_lo, s[18:19], v[34:35]",
            "v_cmp_lt_i32_e32 vcc_lo, s3, v3",
            "v_cmp_le_i32_e32 vcc_lo, s13, v100",
            "v_cmp_gt_i32_e32 vcc_lo, 0x80, v0",
            "v_cmp_ge_i32_e32 vcc_lo, v64, v94",
            "v_cmp_eq_u16_e32 vcc_lo, 0, v19",
            "v_cmp_lt_u32_e32 vcc_lo, s5, v9",
            "v_cmp_eq_u32_e32 vcc_lo, 0, v6",
            "v_cmp_le_u32_e32 vcc_lo, s1, v1",
            "v_cmp_gt_u32_e32 vcc_lo, s14, v5",
            "v_cmp_ne_u32_e32 vcc_lo, 0, v4",
            "v_cmp_ge_u32_e32 vcc_lo, s14, v5",
            "v_cmp_lt_u64_e32 vcc_lo, 16, v[19:20]",
            "v_cmp_eq_u64_e32 vcc_lo, v[2:3], v[6:7]",
            "v_cmp_le_u64_e32 vcc_lo, s[28:29], v[4:5]",
            "v_cmp_ne_u64_e32 vcc_lo, 0, v[3:4]",
            "v_cmp_ge_u64_e32 vcc_lo, s[28:29], v[4:5]",
            "v_cmpx_lt_f32_e32 0, v66",
            "v_cmpx_lt_u64_e32 0x4d2, v[19:20]",
        ];
        test_instructions(&BINARY, &ASM);
    }

    #[test]
    fn test_vop1() {
        const BINARY: [&'static [u32]; 18] = [
            &[0x7E000000],
            &[0x7E5E0504],
            &[0x7E060204],
            &[0x7E240A09],
            &[0x7E060C09],
            &[0x7E060F03],
            &[0x7E001500],
            &[0x7F0A1785],
            &[0x7E0C1F08],
            &[0x7E102102],
            &[0x7E262314],
            &[0x7E044302],
            &[0x7E005500],
            &[0x7E065703],
            &[0x7E1C5F0C],
            &[0x7E4C6322],
            &[0x7E4A6724],
            &[0x7E12722B],
        ];
        const ASM: [&'static str; 18] = [
            "v_nop",
            "v_readfirstlane_b32 s47, v4",
            "v_mov_b32_e32 v3, s4",
            "v_cvt_f32_i32_e32 v18, s9",
            "v_cvt_f32_u32_e32 v3, s9",
            "v_cvt_u32_f32_e32 v3, v3",
            "v_cvt_f16_f32_e32 v0, v0",
            "v_cvt_f32_f16_e32 v133, v133",
            "v_cvt_f32_f64_e32 v6, v[8:9]",
            "v_cvt_f64_f32_e32 v[8:9], v2",
            "v_cvt_f32_ubyte0_e32 v19, v20",
            "v_trunc_f32_e32 v2, v2",
            "v_rcp_f32_e32 v0, v0",
            "v_rcp_iflag_f32_e32 v3, v3",
            "v_rcp_f64_e32 v[14:15], v[12:13]",
            "v_rsq_f64_e32 v[38:39], v[34:35]",
            "v_sqrt_f32_e32 v37, v36",
            "v_ffbh_u32_e32 v9, s43",
        ];
        test_instructions(&BINARY, &ASM);
    }

    #[test]
    fn test_vop2() {
        const BINARY: [&'static [u32]; 30] = [
            &[0x58176B0B, 0x9F000000],
            &[0x5A176B0B, 0x9F000000],
            &[0x020A1105],
            &[0x04012581],
            &[0x062A1B0E],
            &[0x081C1D0D],
            &[0x0A100C0C],
            &[0x10060602],
            &[0x120E8A03],
            &[0x16C402FF, 0x00000080],
            &[0x18020024],
            &[0x1A003511],
            &[0x20101108],
            &[0x220C0D09],
            &[0x277B7A82],
            &[0x2C080690],
            &[0x300E0E9F],
            &[0x340C0487],
            &[0x36060602],
            &[0x38000006],
            &[0x3A020203],
            &[0x4A080504],
            &[0x4C100A80],
            &[0x4E0E0807],
            &[0x50080680],
            &[0x52040405],
            &[0x540C0480],
            &[0x56200D0F],
            &[0x6A305E0C],
            &[0x6C303219],
        ];
        const ASM: [&'static str; 30] = [
            "v_fmamk_f32 v11, v11, 0x9f000000, v181",
            "v_fmaak_f32 v11, v11, v181, 0x9f000000",
            "v_cndmask_b32_e32 v5, v5, v8, vcc_lo",
            "v_dot2c_f32_f16_e32 v0, v129, v146",
            "v_add_f32_e32 v21, v14, v13",
            "v_sub_f32_e32 v14, v13, v14",
            "v_subrev_f32_e32 v8, s12, v6",
            "v_mul_f32_e32 v3, s2, v3",
            "v_mul_i32_i24_e32 v7, s3, v69",
            "v_mul_u32_u24_e32 v98, 0x80, v1",
            "v_mul_hi_u32_u24_e32 v1, s36, v0",
            "v_dot4c_i32_i8_e32 v0, v17, v26",
            "v_max_f32_e32 v8, v8, v8",
            "v_min_i32_e32 v6, v9, v6",
            "v_min_u32_e32 v189, 2, v189",
            "v_lshrrev_b32_e32 v4, 16, v3",
            "v_ashrrev_i32_e32 v7, 31, v7",
            "v_lshlrev_b32_e32 v6, 7, v2",
            "v_and_b32_e32 v3, s2, v3",
            "v_or_b32_e32 v0, s6, v0",
            "v_xor_b32_e32 v1, s3, v1",
            "v_add_nc_u32_e32 v4, v4, v2",
            "v_sub_nc_u32_e32 v8, 0, v5",
            "v_subrev_nc_u32_e32 v7, s7, v4",
            "v_add_co_ci_u32_e32 v4, vcc_lo, 0, v3, vcc_lo",
            "v_sub_co_ci_u32_e32 v2, vcc_lo, s5, v2, vcc_lo",
            "v_subrev_co_ci_u32_e32 v6, vcc_lo, 0, v2, vcc_lo",
            "v_fmac_f32_e32 v16, v15, v6",
            "v_mul_f16_e32 v24, s12, v47",
            "v_fmac_f16_e32 v24, s25, v25",
        ];
        test_instructions(&BINARY, &ASM);
    }

    #[test]
    fn test_smem() {
        const BINARY: [&'static [u32]; 16] = [
            &[0xF4000002, 0xFA000004],
            &[0xF4040003, 0xFA000008],
            &[0xF4080303, 0xFA000000],
            &[0xF40C0202, 0xFA000000],
            &[0xF4100600, 0xFA000008],
            &[0xF4200002, 0xFA000004],
            &[0xF4250002, 0xFA000008],
            &[0xF4284302, 0xFA000000],
            &[0xF42D4202, 0xFA000000],
            &[0xF4300602, 0xFA000008],
            &[0xF47C0000, 0x00000000],
            &[0xF4800000, 0x00000000],
            &[0xF4940080, 0x00000000],
            &[0xF4980101, 0xFA001234],
            &[0xF49C0102, 0x02000000],
            &[0xF40003C4, 0xFA1FFE00],
        ];
        const ASM: [&'static str; 16] = [
            "s_load_dword s0, s[4:5], 0x4",
            "s_load_dwordx2 s[0:1], s[6:7], 0x8",
            "s_load_dwordx4 s[12:15], s[6:7], null",
            "s_load_dwordx8 s[8:15], s[4:5], null",
            "s_load_dwordx16 s[24:39], s[0:1], 0x8",
            "s_buffer_load_dword s0, s[4:7], 0x4",
            "s_buffer_load_dwordx2 s[0:1], s[4:7], 0x8 glc",
            "s_buffer_load_dwordx4 s[12:15], s[4:7], null dlc",
            "s_buffer_load_dwordx8 s[8:15], s[4:7], null glc dlc",
            "s_buffer_load_dwordx16 s[24:39], s[4:7], 0x8",
            "s_gl1_inv",
            "s_dcache_inv",
            "s_memrealtime s[2:3]",
            "s_atc_probe 4, s[2:3], 0x1234",
            "s_atc_probe_buffer 4, s[4:7], s1",
            "s_load_dword s15, s[8:9], -0x200",
        ];
        test_instructions(&BINARY, &ASM);
    }

    #[test]
    fn test_fmt_float() {
        const BINARY: [&'static [u32]; 9] = [
            &[0xBE8003F0],
            &[0xBE8003F1],
            &[0xBE8003F2],
            &[0xBE8003F3],
            &[0xBE8003F4],
            &[0xBE8003F5],
            &[0xBE8003F6],
            &[0xBE8003F7],
            &[0xBE8003F8],
        ];
        const ASM: [&'static str; 9] = [
            "s_mov_b32 s0, 0.5",
            "s_mov_b32 s0, -0.5",
            "s_mov_b32 s0, 1.0",
            "s_mov_b32 s0, -1.0",
            "s_mov_b32 s0, 2.0",
            "s_mov_b32 s0, -2.0",
            "s_mov_b32 s0, 4.0",
            "s_mov_b32 s0, -4.0",
            "s_mov_b32 s0, 0.15915494",
        ];
        test_instructions(&BINARY, &ASM);
    }

    #[test]
    fn test_ldsgds() {
        const BINARY: [&'static [u32]; 34] = [
            &[0xD8340000, 0x00006862],
            &[0xD8380100, 0x00050A02],
            &[0xD83C0200, 0x001C0D0B],
            &[0xD8780000, 0x00002622],
            &[0xD87C0000, 0x0000B6B2],
            &[0xD8D80000, 0x05000016],
            &[0xD8DC0100, 0x07000009],
            &[0xD8E00200, 0x2300001B],
            &[0xD8E40000, 0x38000014],
            &[0xD8E80800, 0x13000013],
            &[0xD8F00000, 0x05000002],
            &[0xD9340000, 0x00000512],
            &[0xD9380100, 0x00110F02],
            &[0xD93C0400, 0x00020020],
            &[0xD9D80000, 0x410000A0],
            &[0xD9DC0100, 0x1300001F],
            &[0xDA800080, 0x00002622],
            &[0xDA840100, 0x0000B6B2],
            &[0xDA9C0020, 0x0A00002C],
            &[0xDB7C2000, 0x00006A78],
            &[0xDBFC0000, 0x6E00004E],
            &[0xD8F8FFC0, 0xBC000000],
            &[0xDACC0000, 0x15000F1B],
            &[0xD8B60004, 0x01000302],
            &[0xD8B81000, 0x01050403],
            &[0xD8000000, 0x00000201],
            &[0xD8800000, 0x01000302],
            &[0xD8400000, 0x00030201],
            &[0xD8C00000, 0x01040302],
            &[0xD8760010, 0x00000001],
            &[0xD8620010, 0x00000000],
            &[0xD8500000, 0x00000000],
            &[0xDAC60020, 0x01000000],
            &[0xDAC20020, 0x00000100],
        ];
        const ASM: [&'static str; 34] = [
            "ds_write_b32 v98, v104",
            "ds_write2_b32 v2, v10, v5 offset1:1",
            "ds_write2st64_b32 v11, v13, v28 offset1:2",
            "ds_write_b8 v34, v38",
            "ds_write_b16 v178, v182",
            "ds_read_b32 v5, v22",
            "ds_read2_b32 v[7:8], v9 offset1:1",
            "ds_read2st64_b32 v[35:36], v27 offset1:2",
            "ds_read_i8 v56, v20",
            "ds_read_u8 v19, v19 offset:2048",
            "ds_read_u16 v5, v2",
            "ds_write_b64 v18, v[5:6]",
            "ds_write2_b64 v2, v[15:16], v[17:18] offset1:1",
            "ds_write2st64_b64 v32, v[0:1], v[2:3] offset1:4",
            "ds_read_b64 v[65:66], v160",
            "ds_read2_b64 v[19:22], v31 offset1:1",
            "ds_write_b8_d16_hi v34, v38 offset:128",
            "ds_write_b16_d16_hi v178, v182 offset:256",
            "ds_read_u16_d16_hi v10, v44 offset:32",
            "ds_write_b128 v120, v[106:109] offset:8192",
            "ds_read_b128 v[110:113], v78",
            "ds_append v188 offset:65472",
            "ds_bpermute_b32 v21, v27, v15",
            "ds_wrxchg_rtn_b32 v1, v2, v3 offset:4 gds",
            "ds_wrxchg2_rtn_b32 v[1:2], v3, v4, v5 offset1:16",
            "ds_add_u32 v1, v2",
            "ds_add_rtn_u32 v1, v2, v3",
            "ds_cmpst_b32 v1, v2, v3",
            "ds_cmpst_rtn_b32 v1, v2, v3, v4",
            "ds_gws_barrier v1 offset:16 gds",
            "ds_gws_sema_release_all offset:16 gds",
            "ds_nop",
            "ds_read_addtid_b32 v1 offset:32 gds",
            "ds_write_addtid_b32 v1 offset:32 gds",
        ];
        test_instructions(&BINARY, &ASM);
    }

    #[test]
    fn test_mubuf() {
        const BINARY: [&'static [u32]; 15] = [
            &[0xE0301000, 0x80046504],
            &[0xE0341000, 0x80023E3C],
            &[0xE0381000, 0x80066600],
            &[0xE0681000, 0x80040085],
            &[0xE06C1000, 0x80040048],
            &[0xE0700004, 0x21000100],
            &[0xE0741000, 0x80050009],
            &[0xE0781000, 0x80023C40],
            &[0xE0801000, 0x80022624],
            &[0xE0841002, 0x80022E24],
            &[0xE0901000, 0x8002B6B4],
            &[0xE0941002, 0x8002B6B4],
            &[0xE0C45000, 0x80054845],
            &[0xE1C40000, 0x00000000],
            &[0xE1C80000, 0x00000000],
        ];
        const ASM: [&'static str; 15] = [
            "buffer_load_dword v101, v4, s[16:19], 0 offen",
            "buffer_load_dwordx2 v[62:63], v60, s[8:11], 0 offen",
            "buffer_load_dwordx4 v[102:105], v0, s[24:27], 0 offen",
            "buffer_store_short v0, v133, s[16:19], 0 offen",
            "buffer_store_short_d16_hi v0, v72, s[16:19], 0 offen",
            "buffer_store_dword v1, off, s[0:3], s33 offset:4",
            "buffer_store_dwordx2 v[0:1], v9, s[20:23], 0 offen",
            "buffer_store_dwordx4 v[60:63], v64, s[8:11], 0 offen",
            "buffer_load_ubyte_d16 v38, v36, s[8:11], 0 offen",
            "buffer_load_ubyte_d16_hi v46, v36, s[8:11], 0 offen offset:2",
            "buffer_load_short_d16 v182, v180, s[8:11], 0 offen",
            "buffer_load_short_d16_hi v182, v180, s[8:11], 0 offen offset:2",
            "buffer_atomic_cmpswap v[72:73], v69, s[20:23], 0 offen glc",
            "buffer_gl0_inv",
            "buffer_gl1_inv",
        ];
        test_instructions(&BINARY, &ASM);
    }

    #[test]
    fn test_scalar_reg_slice() {
        use DecodeOperand::*;
        let operands = Decoder::new(&[])
            .parse_operands(0xfffffff4, &[ScalarRegSlice(0, 6, 2, 1)])
            .expect("Can not decode operands");
        assert!(matches!(
            operands.as_slice(),
            [Operand::ScalarRegister(104, 2)]
        ));
        Decoder::new(&[])
            .parse_operands(0xfc, &[ScalarRegSlice(2, 8, 1, 1)])
            .expect_err("Not an scalar general-purpose register");
    }

    #[test]
    fn test_sdwa() {
        const BINARY: [&[u32]; 11] = [
            &[0x362402F9, 0x86050609],
            &[0x7D5446F9, 0x06808800],
            &[0x7D840AF9, 0x06040000],
            &[0x7D5A38F9, 0x06808505],
            &[0x7E3416F9, 0x0005061A],
            &[0x340006F9, 0x0206061A],
            &[0x2C1E12F9, 0x0606010F],
            &[0x122426F9, 0x08080612],
            &[0x380A08F9, 0x06800616],
            &[0x4A0206F9, 0x08060602],
            &[0x7E02C2F9, 0x00351402],
        ];
        const ASM: [&str; 11] = [
            "v_and_b32_sdwa v18, v9, s1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD",
            "v_cmp_eq_u16_sdwa s8, s0, v35 src0_sel:BYTE_0 src1_sel:DWORD",
            "v_cmp_eq_u32_sdwa vcc_lo, v0, v5 src0_sel:WORD_0 src1_sel:DWORD",
            "v_cmp_ne_u16_sdwa s5, s5, v28 src0_sel:BYTE_0 src1_sel:DWORD",
            "v_cvt_f32_f16_sdwa v26, v26 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1",
            "v_lshlrev_b32_sdwa v0, v26, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_2",
            "v_lshrrev_b32_sdwa v15, v15, v9 dst_sel:BYTE_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
            "v_mul_i32_i24_sdwa v18, sext(v18), sext(v19) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0",
            "v_or_b32_sdwa v5, s22, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD",
            "v_add_nc_u32_sdwa v1, v2, sext(v3) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0",
            "v_cos_f16_sdwa v1, -|v2| dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1",
        ];
        test_instructions(&BINARY, &ASM);
    }

    #[test]
    fn test_vop3b() {
        const BINARY: [&[u32]; 11] = [
            &[0xD70F0001, 0x00020600],
            &[0xD7106A03, 0x00020880],
            &[0xD7196AB7, 0x00036E81],
            &[0xD528020D, 0x000A1411],
            &[0xD5290207, 0x00090080],
            &[0xD52A0047, 0x01AA8C80],
            &[0xD56D0909, 0x04220F07],
            &[0xD56E0D0C, 0x042A0B05],
            &[0xD5760006, 0x00401104],
            &[0xD577010D, 0x043404FF, 0xFFFFFFE0],
            &[0xD70F8001, 0x00020600],
        ];
        const ASM: [&'static str; 11] = [
            "v_add_co_u32 v1, s0, s0, v3",
            "v_sub_co_u32 v3, vcc_lo, 0, v4",
            "v_subrev_co_u32 v183, vcc_lo, 1, v183",
            "v_add_co_ci_u32_e64 v13, s2, s17, v10, s2",
            "v_sub_co_ci_u32_e64 v7, s2, 0, 0, s2",
            "v_subrev_co_ci_u32_e64 v71, s0, 0, v70, vcc_lo",
            "v_div_scale_f32 v9, s9, v7, v7, v8",
            "v_div_scale_f64 v[12:13], s13, v[5:6], v[5:6], v[10:11]",
            "v_mad_u64_u32 v[6:7], s0, v4, s8, s[16:17]",
            "v_mad_i64_i32 v[13:14], s1, 0xffffffe0, s2, v[13:14]",
            "v_add_co_u32 v1, s0, s0, v3 clamp",
        ];
        test_instructions(&BINARY, &ASM);
    }

    #[test]
    fn test_vop3a() {
        const BINARY: [&[u32]; 34] = [
            &[0xD5420028, 0x04000742],
            &[0xD5430000, 0x04041300],
            &[0xD5480006, 0x02090315],
            &[0xD5490043, 0x02410102],
            &[0xD54A0018, 0x041500FF, 0x0000FFFF],
            &[0xD54B0013, 0x044E2B16],
            &[0xD54C0008, 0x03CA1508],
            &[0xD54E0006, 0x0242351B],
            &[0xD5530003, 0x00200F03],
            &[0xD55F0007, 0x04220F09],
            &[0xD5600005, 0x042A0B0C],
            &[0xD5640008, 0x00001D08],
            &[0xD5650208, 0x4002110A],
            &[0xD5690004, 0x0002080A],
            &[0xD56A0006, 0x00001303],
            &[0xD56F0009, 0x042E1509],
            &[0xD570000C, 0x044A1D0C],
            &[0xD5780001, 0x042C5A2F],
            &[0xD6FF0009, 0x00021882],
            &[0xD7000001, 0x0002000A],
            &[0xD7010004, 0x00020C9E],
            &[0xD7110000, 0x00020300],
            &[0xD7440003, 0x00580E80],
            &[0xD7460007, 0x00250D02],
            &[0xD7470008, 0x02060D09],
            &[0xD7600004, 0x00010128],
            &[0xD7610028, 0x00010421],
            &[0xD76D0000, 0x04001001],
            &[0xD76F0001, 0x04051009],
            &[0xD7710002, 0x041C1104],
            &[0xD7720003, 0x04220F06],
            &[0xD77600C2, 0x000387C2],
            &[0xD70D9001, 0x00020702],
            &[0xD5320001, 0x10020702],
        ];
        const ASM: [&'static str; 34] = [
            "v_mad_i32_i24 v40, v66, s3, v0",
            "v_mad_u32_u24 v0, v0, s9, v1",
            "v_bfe_u32 v6, v21, 1, 2",
            "v_bfe_i32 v67, v2, 0, 16",
            "v_bfi_b32 v24, 0xffff, 0, v5",
            "v_fma_f32 v19, v22, v21, v19",
            "v_fma_f64 v[8:9], v[8:9], v[10:11], 1.0",
            "v_alignbit_b32 v6, v27, v26, 16",
            "v_min3_u32 v3, v3, s7, s8",
            "v_div_fixup_f32 v7, v9, v7, v8",
            "v_div_fixup_f64 v[5:6], v[12:13], v[5:6], v[10:11]",
            "v_add_f64 v[8:9], v[8:9], s[14:15]",
            "v_mul_f64 v[8:9], v[10:11], -|v[8:9]|",
            "v_mul_lo_u32 v4, s10, v4",
            "v_mul_hi_u32 v6, v3, s9",
            "v_div_fmas_f32 v9, v9, v10, v11",
            "v_div_fmas_f64 v[12:13], v[12:13], v[14:15], v[18:19]",
            "v_xor3_b32 v1, s47, s45, v11",
            "v_lshlrev_b64 v[9:10], 2, v[12:13]",
            "v_lshrrev_b64 v[1:2], s10, v[0:1]",
            "v_ashrrev_i64 v[4:5], 30, v[6:7]",
            "v_pack_b32_f16 v0, v0, v1",
            "v_perm_b32 v3, 0, s7, s22",
            "v_lshl_add_u32 v7, v2, 6, s9",
            "v_add_lshl_u32 v8, v9, v6, 1",
            "v_readlane_b32 s4, v40, 0",
            "v_writelane_b32 v40, s33, 2",
            "v_add3_u32 v0, s1, s8, v0",
            "v_lshl_or_b32 v1, s9, 8, v1",
            "v_and_or_b32 v2, v4, s8, v7",
            "v_or3_b32 v3, v6, v7, v8",
            "v_sub_nc_i32 v194, v194, v195",
            "v_add_nc_i16 v1, v2, v3 op_sel:[0,1,0] clamp",
            "v_add_f16_e64 v1, v2, v3 mul:4",
        ];
        test_instructions(&BINARY, &ASM);
    }

    #[test]
    fn test_vopx_in_vop3a_format() {
        const BINARY: [&[u32]; 48] = [
            &[0xD5030007, 0x0001E40F],
            &[0xD7030002, 0x00020528],
            &[0xD5250043, 0x00005143],
            &[0xD51B00C0, 0x00017F01],
            &[0xD7080013, 0x0002508F],
            &[0xD51800C5, 0x00003E80],
            &[0xD488006A, 0x0001FF24, 0x00000260],
            &[0xD4A8006A, 0x0001FF0A, 0x00000180],
            &[0xD4CA0001, 0x00010019],
            &[0xD402006A, 0x00010024],
            &[0xD4220004, 0x00024412],
            &[0xD4AA0003, 0x00010003],
            &[0xD4C20000, 0x00021280],
            &[0xD4060000, 0x00025080],
            &[0xD4860001, 0x0002A702],
            &[0xD4C60000, 0x00022716],
            &[0xD4040000, 0x0002130E],
            &[0xD4840000, 0x000202FF, 0x00000080],
            &[0xD4C40002, 0x00021E83],
            &[0xD4830001, 0x0002BE0A],
            &[0xD4C30000, 0x00020E09],
            &[0xD401006A, 0x00010102],
            &[0xD4810000, 0x000200FF, 0x0000007F],
            &[0xD4A90009, 0x00010401],
            &[0xD4C10035, 0x00028541],
            &[0xD4ED000A, 0x0001000C],
            &[0xD40D0000, 0x00023A80],
            &[0xD42D0008, 0x00010014],
            &[0xD4AD0004, 0x00010000],
            &[0xD4C5003C, 0x00029348],
            &[0xD4080001, 0x00021D0E],
            &[0xD5010006, 0x00020D07],
            &[0xD58D000B, 0x2000010B],
            &[0xD76A00C5, 0x000387C2],
            &[0xD52B001A, 0x40026724],
            &[0xD7140004, 0x00002C88],
            &[0xD51A000A, 0x00005709],
            &[0xD7070002, 0x0002268B],
            &[0xD51600BD, 0x0000B882],
            &[0xD7660000, 0x00020005],
            &[0xD7650000, 0x00000805],
            &[0xD50800BD, 0x00004902],
            &[0xD705001E, 0x00023408],
            &[0xD50B0001, 0x0000120C],
            &[0xD51C002D, 0x0001080B],
            &[0xD5B1010A, 0x00000108],
            &[0xD5290207, 0x00090080],
            &[0xD5050071, 0x1802E373],
        ];
        const ASM: [&'static str; 48] = [
            "v_add_f32_e64 v7, s15, 1.0",
            "v_add_nc_u16 v2, v40, v2",
            "v_add_nc_u32_e64 v67, v67, s40",
            "v_and_b32_e64 v192, v1, 63",
            "v_ashrrev_i16 v19, 15, v40",
            "v_ashrrev_i32_e64 v197, 0, s31",
            "v_cmp_class_f32_e64 vcc_lo, v36, 0x260",
            "v_cmp_class_f64_e64 vcc_lo, v[10:11], 0x180",
            "v_cmp_eq_f16_e64 s1, s25, 0",
            "v_cmp_eq_f32_e64 vcc_lo, s36, 0",
            "v_cmp_eq_f64_e64 s4, s[18:19], v[34:35]",
            "v_cmp_eq_u16_e64 s3, s3, 0",
            "v_cmp_eq_u32_e64 s0, 0, v9",
            "v_cmp_ge_f32_e64 s0, 0, v40",
            "v_cmp_ge_i32_e64 s1, v2, v83",
            "v_cmp_ge_u32_e64 s0, v22, v19",
            "v_cmp_gt_f32_e64 s0, v14, v9",
            "v_cmp_gt_i32_e64 s0, 0x80, v1",
            "v_cmp_gt_u32_e64 s2, 3, v15",
            "v_cmp_le_i32_e64 s1, s10, v95",
            "v_cmp_le_u32_e64 s0, s9, v7",
            "v_cmp_lt_f32_e64 vcc_lo, v2, 0",
            "v_cmp_lt_i32_e64 s0, 0x7f, v0",
            "v_cmp_lt_u16_e64 s9, s1, 2",
            "v_cmp_lt_u32_e64 s53, v65, v66",
            "v_cmp_neq_f16_e64 s10, s12, 0",
            "v_cmp_neq_f32_e64 s0, 0, v29",
            "v_cmp_neq_f64_e64 s8, s[20:21], 0",
            "v_cmp_ne_u16_e64 s4, s0, 0",
            "v_cmp_ne_u32_e64 s60, v72, v73",
            "v_cmp_u_f32_e64 s1, v14, v14",
            "v_cndmask_b32_e64 v6, v7, v6, s0",
            "v_cvt_flr_i32_f32_e64 v11, -v11",
            "v_cvt_pk_u16_u32 v197, v194, v195",
            "v_fmac_f32_e64 v26, v36, -v51",
            "v_lshlrev_b16 v4, 8, s22",
            "v_lshlrev_b32_e64 v10, v9, s43",
            "v_lshrrev_b16 v2, 11, v19",
            "v_lshrrev_b32_e64 v189, 2, s92",
            "v_mbcnt_hi_u32_b32 v0, s5, v0",
            "v_mbcnt_lo_u32_b32 v0, s5, s4",
            "v_mul_f32_e64 v189, v2, s36",
            "v_mul_lo_u16 v30, s8, v26",
            "v_mul_u32_u24_e64 v1, s12, s9",
            "v_or_b32_e64 v45, s11, 4",
            "v_rsq_f64_e64 v[10:11], |v[8:9]|",
            "v_sub_co_ci_u32_e64 v7, s2, 0, 0, s2",
            "v_subrev_f32_e64 v113, v115, v113 div:2",
        ];
        test_instructions(&BINARY, &ASM);
    }

    #[test]
    fn test_dpp16() {
        const BINARY: [&[u32]; 6] = [
            &[0x7F7A02FA, 0xFF0055BD],
            &[0x06E8E6FA, 0xFF008073],
            &[0x57756EFA, 0xFF00D874],
            &[0x3B6E02FA, 0xFF00AD01],
            &[0x4A0206FA, 0xFF0D4002],
            &[0x7E0248FA, 0xFF0D0A02],
        ];
        const ASM: [&str; 6] = [
            "v_mov_b32_dpp v189, v189 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf",
            "v_add_f32_dpp v116, v115, v115 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf",
            "v_fmac_f32_dpp v186, v116, v183 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf",
            "v_xor_b32_dpp v183, v1, v1 quad_perm:[1,3,2,2] row_mask:0xf bank_mask:0xf",
            "v_add_nc_u32_dpp v1, v2, v3 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:0 fi:1",
            "v_floor_f32_dpp v1, v2 row_shl:10 row_mask:0xf bank_mask:0xf bound_ctrl:0 fi:1",
        ];
        test_instructions(&BINARY, &ASM);
    }

    #[test]
    fn test_vop3p() {
        const BINARY: [&[u32]; 5] = [
            &[0xCC16C036, 0x1C560F05],
            &[0xCC200000, 0x14031825],
            &[0xCC0E4000, 0x0C029541],
            &[0xCC20100B, 0x1C461951],
            &[0xCC0E5019, 0x1C62BB5B],
        ];
        const ASM: [&str; 5] = [
            "v_dot4_i32_i8 v54, v5, v7, v21 clamp",
            "v_fma_mix_f32 v0, s37, v140, v0 op_sel_hi:[0,1,0]",
            "v_pk_fma_f16 v0, v65, v74, v0 op_sel_hi:[1,0,1]",
            "v_fma_mix_f32 v11, v81, v12, v17 op_sel:[0,1,0] op_sel_hi:[1,1,0]",
            "v_pk_fma_f16 v25, v91, v93, v24 op_sel:[0,1,0]",
        ];
        test_instructions(&BINARY, &ASM);
    }

    #[test]
    fn test_invalid() {
        const BINARY: [&[u32]; 2] = [&[0x00000000], &[0xFFFFFFFF]];
        const ASM: [&str; 2] = [".long 0x00000000", ".long 0xffffffff"];
        test_instructions(&BINARY, &ASM);
    }
}
