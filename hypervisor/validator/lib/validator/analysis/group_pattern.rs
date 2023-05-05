use crate::analysis::GroupPattern::Phi;
use crate::analysis::{InstructionUse, PHIAnalysis};
use crate::ir::instruction::BinaryOperator;
use crate::ir::{APConstant, Function, Value};
use crate::isa::rdna2::isa::Opcode;
use crate::isa::rdna2::opcodes::{SOP2Opcode, VOP2Opcode, VOP3ABOpcode};
use crate::isa::rdna2::RDNA2Target;
use std::iter::zip;

#[derive(Debug, Default)]
pub(crate) struct WideOperand {
    pub(crate) inst_idx: usize,
    pub(crate) lhs_op_idx: usize,
    pub(crate) rhs_op_idx: usize,
}

#[derive(Debug)]
pub(crate) enum GroupPattern {
    // LHS and RHS
    AddSubU64(BinaryOperator, [WideOperand; 2]),
    ShiftU64(BinaryOperator, usize, usize, usize), // inst_idx, lhs_idx, rhs_idx
    // multiply two 32-bit operands and return the 64-bit result
    // Record only the low 32-bit operand for now
    Mul32U64(WideOperand),
    Mad32U64(WideOperand, usize), // operand of mul and the index of the add
    ZExt(Value),
    SExt(Value),
    Shl32(Value),
    PtrMask(Value, Value),
    Phi(usize, usize),
    Constant(APConstant),
    KernelArgumentBase,
    DispatchPacketBase,
}

impl WideOperand {
    fn new(inst_idx: usize, lhs_op_idx: usize, rhs_op_idx: usize) -> WideOperand {
        WideOperand {
            inst_idx,
            lhs_op_idx,
            rhs_op_idx,
        }
    }
}

impl GroupPattern {
    pub(crate) fn detect_pattern<'a>(
        func: &'a Function<'a>,
        du: &'a PHIAnalysis,
        values: &[Value],
    ) -> Option<GroupPattern> {
        Self::detect_pattern_impl(func, du, values, true)
    }

    fn detect_pattern_impl<'a>(
        func: &'a Function<'a>,
        du: &'a PHIAnalysis,
        values: &[Value],
        inspect_swapped_pair: bool,
    ) -> Option<GroupPattern> {
        if values.len() != 2 {
            return None;
        }
        let pattern = match (values[0], values[1]) {
            (Value::Instruction(i0_idx, op0_idx), Value::Instruction(i1_idx, op1_idx)) => {
                let (i0, i1) = (&func.instructions[i0_idx], &func.instructions[i1_idx]);
                match ((i0.op, op0_idx), (i1.op, op1_idx)) {
                    (
                        (Opcode::SOP2(SOP2Opcode::S_ADD_U32), 0),
                        (Opcode::SOP2(SOP2Opcode::S_ADDC_U32), 0),
                    )
                    | (
                        (Opcode::VOP3AB(VOP3ABOpcode::V_ADD_CO_U32), 0),
                        (Opcode::VOP2(VOP2Opcode::V_ADD_CO_CI_U32), 0),
                    )
                    | (
                        (Opcode::VOP3AB(VOP3ABOpcode::V_ADD_CO_U32), 0),
                        (Opcode::VOP3AB(VOP3ABOpcode::V_ADD_CO_CI_U32), 0),
                    ) => {
                        if GroupPattern::is_64bit_add(i0.op, i0_idx, i1_idx, du) {
                            let swapped = if inspect_swapped_pair {
                                let (lo0, lo1, hi0, hi1) = (
                                    du.get_def_use(InstructionUse::op(i0_idx, 2, 0))?,
                                    du.get_def_use(InstructionUse::op(i0_idx, 3, 0))?,
                                    du.get_def_use(InstructionUse::op(i1_idx, 2, 0))?,
                                    du.get_def_use(InstructionUse::op(i1_idx, 3, 0))?,
                                );
                                Self::is_64bit_swapped_pair(func, du, &[lo0, hi1])
                                    || Self::is_64bit_swapped_pair(func, du, &[lo1, hi0])
                            } else {
                                false
                            };
                            let op_idx = if inspect_swapped_pair && swapped {
                                [2, 3, 3, 2]
                            } else {
                                [2, 3, 2, 3]
                            };
                            Some(GroupPattern::AddSubU64(
                                BinaryOperator::Add,
                                [
                                    WideOperand::new(i0_idx, op_idx[0], op_idx[1]),
                                    WideOperand::new(i1_idx, op_idx[2], op_idx[3]),
                                ],
                            ))
                        } else {
                            None
                        }
                    }
                    (
                        (Opcode::SOP2(SOP2Opcode::S_SUB_U32), 0),
                        (Opcode::SOP2(SOP2Opcode::S_SUBB_U32), 0),
                    ) => match du.get_def_use(InstructionUse::op(i1_idx, 1, 0))? {
                        Value::Instruction(inst_scc, 1) if inst_scc == i0_idx => {
                            let op_idx = [2, 3, 2, 3];
                            Some(GroupPattern::AddSubU64(
                                BinaryOperator::Sub,
                                [
                                    WideOperand::new(i0_idx, op_idx[0], op_idx[1]),
                                    WideOperand::new(i1_idx, op_idx[2], op_idx[3]),
                                ],
                            ))
                        }
                        _ => None,
                    },
                    (
                        (Opcode::SOP2(SOP2Opcode::S_MUL_I32), 0),
                        (Opcode::SOP2(SOP2Opcode::S_MUL_HI_U32), 0),
                    )
                    | (
                        (Opcode::SOP2(SOP2Opcode::S_MUL_I32), 0),
                        (Opcode::SOP2(SOP2Opcode::S_MUL_HI_I32), 0),
                    ) => {
                        let lhs0 = du.get_def_use(InstructionUse::op(i0_idx, 1, 0))?;
                        let rhs0 = du.get_def_use(InstructionUse::op(i0_idx, 2, 0))?;
                        let lhs1 = du.get_def_use(InstructionUse::op(i1_idx, 1, 0))?;
                        let rhs1 = du.get_def_use(InstructionUse::op(i1_idx, 2, 0))?;
                        if lhs0 == lhs1 && rhs0 == rhs1 {
                            Some(GroupPattern::Mul32U64(WideOperand::new(i0_idx, 1, 2)))
                        } else {
                            None
                        }
                    }
                    (
                        (Opcode::VOP3AB(VOP3ABOpcode::V_MAD_U64_U32), 0),
                        (Opcode::VOP3AB(VOP3ABOpcode::V_MAD_U64_U32), 1),
                    )
                    | (
                        (Opcode::VOP3AB(VOP3ABOpcode::V_MAD_I64_I32), 0),
                        (Opcode::VOP3AB(VOP3ABOpcode::V_MAD_I64_I32), 1),
                    ) if (i0_idx == i1_idx) => Some(GroupPattern::Mad32U64(
                        WideOperand {
                            inst_idx: i0_idx,
                            lhs_op_idx: 2,
                            rhs_op_idx: 3,
                        },
                        4,
                    )),
                    (
                        (Opcode::SOP2(SOP2Opcode::S_LSHL_B64), 0),
                        (Opcode::SOP2(SOP2Opcode::S_LSHL_B64), 1),
                    )
                    | (
                        (Opcode::SOP2(SOP2Opcode::S_LSHR_B64), 0),
                        (Opcode::SOP2(SOP2Opcode::S_LSHR_B64), 1),
                    )
                    | (
                        (Opcode::VOP3AB(VOP3ABOpcode::V_LSHLREV_B64), 0),
                        (Opcode::VOP3AB(VOP3ABOpcode::V_LSHLREV_B64), 1),
                    )
                    | (
                        (Opcode::VOP3AB(VOP3ABOpcode::V_ASHRREV_I64), 0),
                        (Opcode::VOP3AB(VOP3ABOpcode::V_ASHRREV_I64), 1),
                    ) => (i0_idx == i1_idx).then_some({
                        let (lhs, rhs) = match i0.op {
                            Opcode::VOP3AB(_) => (2, 1),
                            _ => (2, 3),
                        };
                        match i0.op {
                            Opcode::SOP2(SOP2Opcode::S_LSHL_B64)
                            | Opcode::VOP3AB(VOP3ABOpcode::V_LSHLREV_B64) => {
                                GroupPattern::ShiftU64(BinaryOperator::LShift, i0_idx, lhs, rhs)
                            }
                            _ => GroupPattern::ShiftU64(BinaryOperator::AShr, i0_idx, lhs, rhs),
                        }
                    }),
                    _ => None,
                }
            }
            (Value::Phi(p0_idx), Value::Phi(p1_idx)) => {
                let (p0, p1) = (&du.phis[p0_idx], &du.phis[p1_idx]);
                // Over conservative analysis for grouped PHIs
                if p0.bb_idx != p1.bb_idx
                    || p0.values.len() != p1.values.len()
                    || p0.values.len() != 2
                    || zip(p0.values.iter(), p1.values.iter()).any(|(x, y)| x.0 != y.0)
                {
                    None
                } else {
                    Some(GroupPattern::Phi(p0_idx, p1_idx))
                }
            }
            (
                Value::Constant(APConstant::ConstantInt(i0)),
                Value::Constant(APConstant::ConstantInt(i1)),
            ) => {
                let v = ((i0 as u32) as isize) | (((i1 as u32) as isize) << 32);
                Some(GroupPattern::Constant(APConstant::ConstantInt(v)))
            }
            (Value::Argument(RDNA2Target::TAG_SGPR_KERNARG_SEGMENT_PTR), Value::Argument(y))
                if y == RDNA2Target::TAG_SGPR_KERNARG_SEGMENT_PTR + 1 =>
            {
                Some(GroupPattern::KernelArgumentBase)
            }
            (Value::Argument(RDNA2Target::TAG_SGPR_DISPATCH_PTR), Value::Argument(y))
                if y == RDNA2Target::TAG_SGPR_DISPATCH_PTR + 1 =>
            {
                Some(GroupPattern::DispatchPacketBase)
            }
            _ => None,
        };
        if pattern.is_some() {
            return pattern;
        }
        // match pattern for zero / signed extension and pointer-mask at last
        match (values[0], values[1]) {
            (val_lo, Value::Instruction(inst_hi_idx, op_hi_idx))
                if matches!(
                    (func.instructions[inst_hi_idx].op, op_hi_idx),
                    (Opcode::SOP2(SOP2Opcode::S_AND_B32), 0),
                ) && du.get_def_use(InstructionUse::op(inst_hi_idx, 3, 0))
                    == Some(Value::Constant(APConstant::ConstantInt(0xffff))) =>
            {
                // group(val_lo, s_and_b32(val_hi, 0xffff)) ==> and(0xffff_ffff_ffff, group(val_lo, val_hi))
                let val_hi = du.get_def_use(InstructionUse::op(inst_hi_idx, 2, 0))?;
                Some(GroupPattern::PtrMask(val_lo, val_hi))
            }
            (Value::Constant(APConstant::ConstantInt(0)), value_hi) => {
                Some(GroupPattern::Shl32(value_hi))
            }
            (value_lo, Value::Constant(APConstant::ConstantInt(0))) => {
                Some(GroupPattern::ZExt(value_lo))
            }
            (value_lo, Value::Instruction(i_idx, op_idx)) => {
                let inst = &func.instructions[i_idx];
                let (src_idx, shift_idx) = match (inst.op, op_idx) {
                    (Opcode::VOP2(VOP2Opcode::V_ASHRREV_I32), 0)
                    | (Opcode::VOP3AB(VOP3ABOpcode::V_ASHRREV_I32), 0) => Some((2, 1)),
                    (Opcode::SOP2(SOP2Opcode::S_ASHR_I32), 0) => Some((2, 3)),
                    _ => None,
                }?;
                let shift = du.get_def_use(InstructionUse::op(i_idx, shift_idx, 0))?;
                let src = du.get_def_use(InstructionUse::op(i_idx, src_idx, 0))?;
                (matches!(shift, Value::Constant(APConstant::ConstantInt(31))) && src == value_lo)
                    .then_some(GroupPattern::SExt(value_lo))
            }
            _ => None,
        }
    }

    fn is_64bit_add(op0: Opcode, i0_idx: usize, i1_idx: usize, du: &PHIAnalysis) -> bool {
        let scc_idx = match op0 {
            Opcode::SOP2(SOP2Opcode::S_ADD_U32) => 1,
            _ => 4,
        };
        matches!(du.get_def_use(InstructionUse::op(i1_idx, scc_idx, 0)),
            Some(Value::Instruction(inst_scc, _)) if inst_scc == i0_idx
        )
    }

    /**
     * Heuristics to detect the compiler that swapped the lo / hi operands
     * when generating 64-bit add.
     *
     * Do NOT consider two constants as swapped for now
     **/
    fn is_64bit_swapped_pair<'a>(
        func: &'a Function<'a>,
        du: &'a PHIAnalysis,
        values: &[Value],
    ) -> bool {
        matches!(
            Self::detect_pattern_impl(func, du, values, false),
            Some(GroupPattern::KernelArgumentBase)
                | Some(GroupPattern::DispatchPacketBase)
                | Some(Phi(_, _))
                | Some(GroupPattern::AddSubU64(_, _))
                | Some(GroupPattern::ShiftU64(_, _, _, _))
                | Some(GroupPattern::Mul32U64(_))
                | Some(GroupPattern::SExt(_))
                | Some(GroupPattern::Mad32U64(_, _))
        )
    }
}
