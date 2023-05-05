use crate::error::{DecodeError, Error, Result};
use crate::fileformat::parser::KernelInfo;
use crate::fileformat::AMDGPUELF;
use crate::ir::instruction::IRInstruction;
use crate::ir::{BasicBlock, Function, Module};
use crate::isa::rdna2;
use crate::isa::rdna2::isa::Opcode;
use crate::isa::rdna2::opcodes::SOPPOpcode;
use crate::isa::rdna2::{Decoder, Instruction};
use smallvec::{smallvec, SmallVec};
use std::collections::{BTreeMap, HashMap};
use std::ops::Range;

const NEXT_INST_REL_OFFSET: i32 = 1;

#[allow(dead_code)]
pub fn disassemble_kernel<'a>(
    blob: &'a [u8],
    name: &str,
) -> Result<(KernelInfo<'a>, Function<'a>)> {
    let elf = AMDGPUELF::parse(blob)?;
    let ki = elf
        .kernels
        .into_iter()
        .find(|k| k.name == name)
        .ok_or_else(|| {
            Error::ELFError(goblin::error::Error::Malformed(
                "Kernel not found".to_string(),
            ))
        })?;
    let f = Disassembler::parse_kernel(&ki)?;
    Ok((ki, f))
}

pub fn disassemble(blob: &[u8]) -> Result<Module> {
    let elf = AMDGPUELF::parse(blob)?;
    let mut kernels = Vec::new();
    elf.kernels.into_iter().try_for_each(|x| {
        let f = Disassembler::parse_kernel(&x)?;
        kernels.push((x, f));
        Ok::<_, Error>(())
    })?;
    Ok(Module { kernels })
}

pub(crate) struct Disassembler {}

impl Disassembler {
    pub(crate) fn parse_kernel<'a>(k: &KernelInfo<'a>) -> Result<Function<'a>> {
        let mut insts = Vec::new();
        let mut inst_pc = Vec::new();
        let mut pc_to_inst = HashMap::new();

        let mut decoder = Decoder::new(k.code);
        decoder.try_for_each(|(pc, inst)| {
            if matches!(inst.op, Opcode::INVALID(_)) {
                return Err(Error::DecodeError(DecodeError::InvalidInstruction));
            }
            inst_pc.push(pc);
            pc_to_inst.insert(pc, insts.len());
            insts.push(inst);
            Ok(())
        })?;

        // Outgoing edges form branches. Note that it has to be augmented with the fallthrough edges.
        let mut branch_targets: HashMap<i32, SmallVec<[i32; 2]>> = HashMap::new();
        // Incoming edges from branches. Also record the start of the BB
        let mut bb_incoming_edges: BTreeMap<i32, SmallVec<[i32; 2]>> = BTreeMap::new();
        bb_incoming_edges.insert(0, SmallVec::new());

        // Find the start of the BB
        insts
            .iter()
            .enumerate()
            .filter_map(|(idx, inst)| Self::maybe_jump_target(inst).map(|x| (idx, x)))
            .try_for_each(|(idx, rel_pc)| {
                let pc = inst_pc[idx];
                let abs_targets: SmallVec<[i32; 2]> = rel_pc
                    .iter()
                    .filter_map(|x| {
                        let abs_pc = (pc as i32 + x) as usize;
                        pc_to_inst.get(&abs_pc).map(|y| *y as i32)
                    })
                    .collect();
                if abs_targets.len() != rel_pc.len() {
                    return Err(Error::DecodeError(DecodeError::InvalidOperand));
                }
                for target_idx in &abs_targets {
                    if let Some(e) = bb_incoming_edges.get_mut(target_idx) {
                        e.push(idx as i32);
                    } else {
                        bb_incoming_edges.insert(*target_idx, smallvec![idx as i32]);
                    }
                }
                branch_targets.insert(idx as i32, abs_targets);
                Ok(())
            })?;

        // Find the start of unreachable BB
        insts
            .iter()
            .enumerate()
            .filter_map(|(idx, inst)| {
                if let Some(IRInstruction::BranchInst(b)) = rdna2::Instruction::wrap(inst.clone()) {
                    if !b.fall_through() {
                        return Some(idx);
                    }
                }
                None
            })
            .try_for_each(|idx| -> Result<()> {
                if idx + 1 == insts.len() {
                    // That's the last instruction of the kernel. Donnot genetare an extra basic block.
                    return Ok(());
                }
                let abs_pc = inst_pc[idx] + NEXT_INST_REL_OFFSET as usize;
                let abs_target = pc_to_inst
                    .get(&abs_pc)
                    .map(|x| *x as i32)
                    .ok_or(Error::DecodeError(DecodeError::InvalidOperand))?;
                if bb_incoming_edges.get(&abs_target).is_none() {
                    bb_incoming_edges.insert(abs_target, smallvec![-1]);
                }
                Ok(())
            })?;

        let mut bbs = Vec::new();
        let last = &[insts.len() as i32];
        let mut it = bb_incoming_edges.iter().map(|x| x.0).chain(last);
        let mut prev = *it.next().unwrap();
        for end in it {
            bbs.push(BasicBlock {
                offset: prev as usize,
                predecessors: SmallVec::new(),
                successors: SmallVec::new(),
                instructions: Range {
                    start: prev as usize,
                    end: *end as usize,
                },
            });
            prev = *end;
        }

        let branch_successors = branch_targets
            .into_iter()
            .map(|(inst_idx, target_offsets)| {
                let target_bb = target_offsets.into_iter().map(|target_offset| {
                    let u = target_offset as usize;
                    bbs.iter()
                        .enumerate()
                        .find(|(_, bb)| bb.instructions.contains(&u))
                        .expect("Invalid jump target")
                        .0
                });
                (inst_idx, target_bb.collect::<SmallVec<[usize; 2]>>())
            })
            .collect::<HashMap<i32, SmallVec<[usize; 2]>>>();

        // Populate successors
        let total_num_bb = bbs.len();
        let mut predecessors: HashMap<usize, SmallVec<[usize; 4]>> = HashMap::new();
        for (bb_idx, bb) in bbs.iter_mut().enumerate() {
            let last_inst_idx = bb.instructions.end as i32 - 1;
            let last_inst = &insts[last_inst_idx as usize];
            let mut succ: SmallVec<[usize; 4]> = SmallVec::new();
            if Self::may_fallthrough(last_inst) && bb_idx + 1 < total_num_bb {
                succ.push(bb_idx + 1);
            }
            if let Some(x) = branch_successors.get(&last_inst_idx) {
                succ.extend_from_slice(x.as_slice());
            }
            succ.sort();
            succ.dedup();
            succ.iter().for_each(|target_bb_idx| {
                if let Some(p) = predecessors.get_mut(target_bb_idx) {
                    p.push(bb_idx);
                } else {
                    predecessors.insert(*target_bb_idx, smallvec![bb_idx]);
                }
            });
            bb.successors = succ;
        }

        predecessors.into_iter().for_each(|(idx, p)| {
            let bb = &mut bbs[idx];
            bb.predecessors = p;
        });

        Ok(Function {
            name: k.name,
            basic_blocks: bbs,
            instructions: insts,
        })
    }

    fn maybe_jump_target(inst: &Instruction) -> Option<SmallVec<[i32; 2]>> {
        if let Some(IRInstruction::BranchInst(b)) = rdna2::Instruction::wrap(inst.clone()) {
            let rel_offset = b.get_rel_target();
            if rel_offset == 0 || !b.fall_through() {
                Some(smallvec!(rel_offset + NEXT_INST_REL_OFFSET))
            } else {
                Some(smallvec!(
                    rel_offset + NEXT_INST_REL_OFFSET,
                    NEXT_INST_REL_OFFSET
                ))
            }
        } else {
            None
        }
    }

    fn may_fallthrough(inst: &Instruction) -> bool {
        !matches!(inst.op, Opcode::SOPP(op) if op == SOPPOpcode::S_BRANCH)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fileformat::SGPRSetup;
    use std::ptr;

    #[test]
    fn test_single_branch() {
        // v_cmp_ne_u32_e32 vcc_lo, 0, v0
        // s_and_saveexec_b32 s4, vcc_lo
        // s_cbranch_execz 0
        // s_endpgm
        const CODE: [u32; 4] = [0x7D8A0080, 0xBE843C6A, 0xBF880000, 0xBF810000];
        let ki = KernelInfo {
            name: "foo",
            code: &CODE,
            pgmrsrcs: [0, 0, 0],
            setup: SGPRSetup::SGPR_KERNARG_SEGMENT_PTR,
            kern_arg_size: 0,
            kern_arg_segment_align: 0,
            group_segment_fixed_size: 0,
            private_segment_fixed_size: 0,
            arguments: vec![],
        };
        let func = Disassembler::parse_kernel(&ki).expect("Failed to parse the function");
        assert_eq!(2, func.basic_blocks.len());
        let (b0, b1) = (&func.basic_blocks[0], &func.basic_blocks[1]);
        assert_eq!(0, b0.predecessors.len());
        assert_eq!(1, b0.successors.len());
        assert!(ptr::eq(b1, b0.get_successors(&func, 0)));
        assert_eq!(3, b0.instructions(&func).len());
        assert_eq!(1, b1.predecessors.len());
        assert!(ptr::eq(b0, b1.get_predecessors(&func, 0)));
        assert_eq!(0, b1.successors.len());
        assert_eq!(1, b1.instructions(&func).len());
    }

    #[test]
    fn test_backedge_fallthrough() {
        // v_cmp_ne_u32_e32 vcc_lo, 0, v0
        // s_and_saveexec_b32 s4, vcc_lo
        // s_cbranch_execz 65534
        // s_endpgm
        const CODE: [u32; 4] = [0x7d8a0080, 0xbe843c6a, 0xbf88fffe, 0xbf810000];
        let ki = KernelInfo {
            name: "foo",
            code: &CODE,
            pgmrsrcs: [0, 0, 0],
            setup: Default::default(),
            kern_arg_size: 0,
            kern_arg_segment_align: 0,
            group_segment_fixed_size: 0,
            private_segment_fixed_size: 0,
            arguments: vec![],
        };
        let func = Disassembler::parse_kernel(&ki).expect("Failed to parse the function");
        let b = &func.basic_blocks;
        assert_eq!(3, b.len());
        assert_eq!(
            [1, 2, 1],
            (0..b.len())
                .map(|x| b[x].instructions(&func).len())
                .collect::<Vec<usize>>()
                .as_slice()
        );
        assert_eq!(
            [0, 2, 1],
            (0..b.len())
                .map(|x| b[x].predecessors.len())
                .collect::<Vec<usize>>()
                .as_slice()
        );
        assert!(ptr::eq(&b[1], b[0].get_successors(&func, 0)));
    }

    #[test]
    fn test_nested_loop() {
        let (_, func) = crate::tests::cfg::nested_loop();
        let b = &func.basic_blocks;
        assert_eq!(5, b.len());
    }

    #[test]
    fn test_unconditional_branch() {
        // s_cmp_eq_i32 s0, 0
        // s_cbranch_scc0 1
        // s_branch 1
        // s_nop 0
        // s_endpgm
        const CODE: [u32; 5] = [0xBF008000, 0xBF840001, 0xBF820001, 0xBF800000, 0xBF810000];
        let ki = KernelInfo {
            name: "foo",
            code: &CODE,
            pgmrsrcs: [0, 0, 0],
            setup: SGPRSetup::SGPR_KERNARG_SEGMENT_PTR,
            kern_arg_size: 0,
            kern_arg_segment_align: 0,
            group_segment_fixed_size: 0,
            private_segment_fixed_size: 0,
            arguments: vec![],
        };
        let func = Disassembler::parse_kernel(&ki).expect("Failed to parse the function");
        let b = &func.basic_blocks;
        assert_eq!(4, b.len());
        assert_eq!(
            [2, 1, 1, 1],
            b.iter()
                .map(|x| x.instructions(&func).len())
                .collect::<Vec<usize>>()
                .as_slice()
        );
        assert_eq!(
            [0, 1, 1, 2],
            b.iter()
                .map(|x| x.predecessors.len())
                .collect::<Vec<usize>>()
                .as_slice()
        );
        assert_eq!(
            [2, 1, 1, 0],
            b.iter()
                .map(|x| x.successors.len())
                .collect::<Vec<usize>>()
                .as_slice()
        );
        assert!(ptr::eq(&b[0], b[1].get_predecessors(&func, 0)));
        assert!(ptr::eq(&b[0], b[2].get_predecessors(&func, 0)));
        assert!(ptr::eq(&b[3], b[1].get_successors(&func, 0)));
        assert!(ptr::eq(&b[3], b[2].get_successors(&func, 0)));
    }

    #[test]
    fn test_nop_inst() {
        const CODE: [u32; 5] = [
            // BB0:
            0xbf840003, // s_cbranch_scc0 BB2
            // BB1:
            0xbf800000, // s_nop 0
            0xbf82fffd, // s_branch BB0
            // BB2:unreachable
            0xbf800000, // s_nop 0
            // BB3:
            0xBF810000, // s_endpgm
        ];
        // 0 -> 1, 3
        // 1 -> 0
        // 2 -> 3
        // 3 -> -
        let ki = KernelInfo {
            name: "",
            code: &CODE,
            pgmrsrcs: [0, 0, 0],
            setup: SGPRSetup::SGPR_KERNARG_SEGMENT_PTR,
            kern_arg_size: 0,
            kern_arg_segment_align: 0,
            group_segment_fixed_size: 0,
            private_segment_fixed_size: 0,
            arguments: vec![],
        };
        let func = Disassembler::parse_kernel(&ki).expect("Failed to parse the function");
        let b = &func.basic_blocks;
        assert_eq!(4, b.len());
        assert_eq!(
            [1, 2, 1, 1],
            b.iter()
                .map(|x| x.instructions(&func).len())
                .collect::<Vec<usize>>()
                .as_slice()
        );
        assert_eq!(
            [1, 1, 0, 2],
            b.iter()
                .map(|x| x.predecessors.len())
                .collect::<Vec<usize>>()
                .as_slice()
        );
        assert_eq!(
            [2, 1, 1, 0],
            b.iter()
                .map(|x| x.successors.len())
                .collect::<Vec<usize>>()
                .as_slice()
        );
        assert!(ptr::eq(&b[1], b[0].get_predecessors(&func, 0)));
        assert!(ptr::eq(&b[0], b[1].get_predecessors(&func, 0)));
        assert!(ptr::eq(&b[0], b[3].get_predecessors(&func, 0)));
        assert!(ptr::eq(&b[2], b[3].get_predecessors(&func, 1)));
        assert!(ptr::eq(&b[0], b[1].get_successors(&func, 0)));
        assert!(ptr::eq(&b[1], b[0].get_successors(&func, 0)));
        assert!(ptr::eq(&b[3], b[0].get_successors(&func, 1)));
        assert!(ptr::eq(&b[3], b[2].get_successors(&func, 0)));
    }
}
