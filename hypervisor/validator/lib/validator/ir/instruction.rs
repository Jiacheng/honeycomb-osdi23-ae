use crate::analysis::{InstructionUse, PHIAnalysis, VirtualUse};
use crate::ir::Type;
use crate::isa::rdna2::isa::{Opcode, Operand};
use crate::isa::rdna2::opcodes::{
    SMEMOpcode, SOP2Opcode, SOPCOpcode, SOPKOpcode, SOPPOpcode, VMEMOpcode, VOP2Opcode,
    VOP3ABOpcode, VOPCOpcode,
};
use crate::isa::rdna2::{Instruction, InstructionModifier};

/**
 * A wrapper for low-leve instruction that facilitates the analysis.
 **/
#[derive(Clone)]
pub enum IRInstruction {
    CmpInst(CmpInst),
    BranchInst(BranchInst),
    StoreInst(StoreInst),
    LoadInst(LoadInst),
    BinOpInst(BinOpInst),
    AtomicInst(AtomicInst),
    TernaryOpInst(TenaryOpInst),
    // Conditional move including min / max / cselect / cndmask
    SelectInst(SelectInst),
}

impl IRInstruction {
    pub fn as_cmp_inst(&self) -> Option<&CmpInst> {
        match self {
            IRInstruction::CmpInst(x) => Some(x),
            _ => None,
        }
    }

    pub fn as_branch_inst(&self) -> Option<&BranchInst> {
        match self {
            IRInstruction::BranchInst(x) => Some(x),
            _ => None,
        }
    }

    pub(crate) fn as_binary_op(&self) -> Option<&BinOpInst> {
        match self {
            IRInstruction::BinOpInst(x) => Some(x),
            _ => None,
        }
    }
}

#[derive(Clone)]
pub struct BranchInst {
    pub(crate) inst: Instruction,
}

#[derive(Clone)]
pub struct CmpInst {
    pub(crate) inst: Instruction,
}

#[derive(Clone)]
pub struct StoreInst {
    pub(crate) inst: Instruction,
}

#[derive(Clone)]
pub struct AtomicInst {
    pub(crate) inst: Instruction,
}

#[derive(Clone)]
pub struct LoadInst {
    pub(crate) inst: Instruction,
}

#[derive(Clone)]
pub struct BinOpInst {
    pub(crate) inst: Instruction,
}

#[derive(Clone)]
pub struct TenaryOpInst {
    pub(crate) inst: Instruction,
}

#[derive(Clone)]
pub struct SelectInst {
    pub(crate) inst: Instruction,
}

/**
 * MemAccessAddress describes the address of memory instructions.
 **/
#[derive(Clone, Debug)]
pub(crate) struct MemAccessAddress {
    pub(crate) base: VirtualUse,
    pub(crate) reg_offset: Option<VirtualUse>,
    pub(crate) imm_offset: Option<VirtualUse>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BinaryOperator {
    Add,
    Sub,
    Mul,
    LShift,
    AShr,
    Min,
    Max,
    And,
    Or,
    Xor,
}

/*
 * BinOpDescriptor provides mappings of the operands of the binary operator
 */
#[derive(Copy, Clone, Debug)]
pub(crate) struct BinOpOperandMapping {
    pub(crate) op: BinaryOperator,
    pub(crate) dst_idx: usize,
    pub(crate) lhs_op_idx: usize,
    pub(crate) rhs_op_idx: usize,
}

impl BinOpOperandMapping {
    const fn new(
        op: BinaryOperator,
        dst_idx: usize,
        lhs_op_idx: usize,
        rhs_op_idx: usize,
    ) -> BinOpOperandMapping {
        BinOpOperandMapping {
            op,
            dst_idx,
            lhs_op_idx,
            rhs_op_idx,
        }
    }
}

/*
 * Ternary instruction dst = (src0 op1 src1) op2 src2
 */
#[derive(Copy, Clone, Debug)]
pub(crate) struct TernaryOperandMapping {
    pub(crate) op0: BinaryOperator,
    pub(crate) op1: BinaryOperator,
    pub(crate) dst_idx: usize,
    pub(crate) src0: usize,
    pub(crate) src1: usize,
    pub(crate) src2: usize,
}

impl TernaryOperandMapping {
    const fn new(
        op0: BinaryOperator,
        op1: BinaryOperator,
        dst_idx: usize,
        src0: usize,
        src1: usize,
        src2: usize,
    ) -> TernaryOperandMapping {
        TernaryOperandMapping {
            op0,
            op1,
            dst_idx,
            src0,
            src1,
            src2,
        }
    }
}

/*
 * CSelect instruction dst = cond ? src0 : src1
 */
#[derive(Copy, Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct CSelectOperandMapping {
    pub(crate) dst_idx: usize,
    pub(crate) selector_idx: usize,
    pub(crate) src0_idx: usize,
    pub(crate) src1_idx: usize,
}

impl CSelectOperandMapping {
    const fn new(
        dst_idx: usize,
        selector_idx: usize,
        src0_idx: usize,
        src1_idx: usize,
    ) -> CSelectOperandMapping {
        CSelectOperandMapping {
            dst_idx,
            selector_idx,
            src0_idx,
            src1_idx,
        }
    }
}

impl BranchInst {
    pub fn new(inst: Instruction) -> Self {
        Self { inst }
    }

    pub fn fall_through(&self) -> bool {
        !matches!(self.inst.op, Opcode::SOPP(SOPPOpcode::S_BRANCH))
    }

    pub fn get_rel_target(&self) -> i32 {
        use SOPPOpcode::*;
        let op = if let Opcode::SOPP(opcode) = self.inst.op {
            match opcode {
                S_BRANCH => Some(self.inst.get_operands()[1]),
                S_CBRANCH_SCC0
                | S_CBRANCH_SCC1
                | S_CBRANCH_VCCZ
                | S_CBRANCH_VCCNZ
                | S_CBRANCH_EXECZ
                | S_CBRANCH_EXECNZ
                | S_CBRANCH_CDBGSYS
                | S_CBRANCH_CDBGUSER
                | S_CBRANCH_CDBGSYS_OR_USER
                | S_CBRANCH_CDBGSYS_AND_USER => Some(self.inst.get_operands()[2]),
                _ => None,
            }
        } else {
            None
        };

        if let Some(Operand::Constant(target)) = op {
            target
        } else {
            unreachable!();
        }
    }
}

impl CmpInst {
    pub fn new(inst: Instruction) -> CmpInst {
        Self { inst }
    }

    pub fn get_type(&self) -> Type {
        match self.inst.op {
            Opcode::SOPK(opcode) => {
                use SOPKOpcode::*;
                match opcode {
                    S_CMPK_EQ_I32 | S_CMPK_LG_I32 | S_CMPK_GT_I32 | S_CMPK_GE_I32
                    | S_CMPK_LT_I32 | S_CMPK_LE_I32 => Type::Int32,
                    S_CMPK_EQ_U32 | S_CMPK_LG_U32 | S_CMPK_GT_U32 | S_CMPK_GE_U32
                    | S_CMPK_LT_U32 | S_CMPK_LE_U32 => Type::Int32,
                    _ => Type::Unknown,
                }
            }
            Opcode::SOPC(opcode) => {
                use SOPCOpcode::*;
                match opcode {
                    S_CMP_EQ_I32 | S_CMP_LG_I32 | S_CMP_GT_I32 | S_CMP_GE_I32 | S_CMP_LT_I32
                    | S_CMP_LE_I32 => Type::Int32,
                    S_CMP_EQ_U32 | S_CMP_LG_U32 | S_CMP_GT_U32 | S_CMP_GE_U32 | S_CMP_LT_U32
                    | S_CMP_LE_U32 | S_BITCMP0_B32 | S_BITCMP1_B32 => Type::Int32,
                    S_BITCMP0_B64 | S_BITCMP1_B64 | S_CMP_EQ_U64 | S_CMP_LG_U64 => Type::Int64,
                }
            }
            Opcode::VOPC(opcode) => Self::type_from_vopc(opcode),
            Opcode::VOP3AB(opcode) => Self::type_from_vopc(opcode.as_vopc().unwrap()),
            _ => Type::Unknown,
        }
    }

    fn type_from_vopc(opcode: VOPCOpcode) -> Type {
        if let Some((ty, sz)) = opcode.get_type_size() {
            use crate::isa::rdna2::CmpDataSize::*;
            use crate::isa::rdna2::CmpInstType::*;
            match (ty, sz) {
                (SignedComparison, B32) | (UnsignedComparison, B32) => Type::Int32,
                (SignedComparison, B64) | (UnsignedComparison, B64) => Type::Int64,
                (FloatComparion, B32) => Type::Float32,
                (FloatComparion, B64) => Type::Float64,
                _ => Type::Unknown,
            }
        } else {
            Type::Unknown
        }
    }
}

impl StoreInst {
    pub fn new(inst: Instruction) -> Self {
        Self { inst }
    }

    pub(crate) fn get_addr<'a>(
        &self,
        inst_idx: usize,
        du: &'a PHIAnalysis<'a>,
    ) -> Option<MemAccessAddress> {
        MemAccessAddress::get_addr(&self.inst, inst_idx, du)
    }

    pub(crate) fn get_dst_dwords(&self) -> usize {
        MemAccessAddress::get_dst_dwords(&self.inst)
    }
}

impl LoadInst {
    pub(crate) fn new(inst: Instruction) -> Self {
        Self { inst }
    }

    pub(crate) fn get_addr(&self, inst_idx: usize, du: &PHIAnalysis) -> Option<MemAccessAddress> {
        MemAccessAddress::get_addr(&self.inst, inst_idx, du)
    }

    pub(crate) fn get_dst_dwords(&self) -> usize {
        MemAccessAddress::get_dst_dwords(&self.inst)
    }
}

impl AtomicInst {
    pub fn new(inst: Instruction) -> Self {
        Self { inst }
    }

    pub(crate) fn get_addr<'a>(
        &self,
        inst_idx: usize,
        du: &'a PHIAnalysis<'a>,
    ) -> Option<MemAccessAddress> {
        MemAccessAddress::get_addr(&self.inst, inst_idx, du)
    }

    #[allow(dead_code)]
    pub(crate) fn get_dst_dwords(&self) -> usize {
        MemAccessAddress::get_dst_dwords(&self.inst)
    }
}

impl MemAccessAddress {
    fn get_dst_dwords(inst: &Instruction) -> usize {
        match inst.op {
            Opcode::SMEM(opcode) => opcode.data_size(),
            Opcode::VMEM(opcode) => opcode.data_size().0,
            _ => unreachable!(),
        }
    }

    fn get_addr(inst: &Instruction, inst_idx: usize, du: &PHIAnalysis) -> Option<MemAccessAddress> {
        use Opcode::{SMEM, VMEM};
        use SMEMOpcode::*;
        use VMEMOpcode::*;

        match inst.op {
            // not supported yet
            VMEM(GLOBAL_LOAD_DWORD_ADDTID) | VMEM(GLOBAL_STORE_DWORD_ADDTID) => None,
            VMEM(_) => {
                let modifier = match &inst.modifier {
                    Some(InstructionModifier::VMem(m)) => Some(m),
                    _ => None,
                }?;
                let offset = VirtualUse::Value(du.get_def_use(InstructionUse::op(inst_idx, 4, 0))?);
                if modifier.saddr_null() == 0 {
                    let vaddr =
                        VirtualUse::Value(du.get_def_use(InstructionUse::op(inst_idx, 1, 0))?);
                    let saddr = VirtualUse::Group(vec![
                        du.get_def_use(InstructionUse::op(inst_idx, 3, 0))?,
                        du.get_def_use(InstructionUse::op(inst_idx, 3, 1))?,
                    ]);
                    Some(MemAccessAddress {
                        base: saddr,
                        reg_offset: Some(vaddr),
                        imm_offset: Some(offset),
                    })
                } else {
                    let vaddr = VirtualUse::Group(vec![
                        du.get_def_use(InstructionUse::op(inst_idx, 1, 0))?,
                        du.get_def_use(InstructionUse::op(inst_idx, 1, 1))?,
                    ]);
                    Some(MemAccessAddress {
                        base: vaddr,
                        reg_offset: None,
                        imm_offset: Some(offset),
                    })
                }
            }
            SMEM(S_LOAD_DWORD)
            | SMEM(S_LOAD_DWORDX2)
            | SMEM(S_LOAD_DWORDX4)
            | SMEM(S_LOAD_DWORDX8)
            | SMEM(S_LOAD_DWORDX16) => {
                let sbase = VirtualUse::Group(vec![
                    du.get_def_use(InstructionUse::op(inst_idx, 1, 0))?,
                    du.get_def_use(InstructionUse::op(inst_idx, 1, 1))?,
                ]);
                let imm_offset =
                    VirtualUse::Value(du.get_def_use(InstructionUse::op(inst_idx, 5, 0))?);
                let soffset = &inst.get_operands()[4];
                let reg_offset = if let Operand::ScalarRegister(_, _) = soffset {
                    Some(VirtualUse::Value(
                        du.get_def_use(InstructionUse::op(inst_idx, 5, 0))?,
                    ))
                } else {
                    None
                };
                Some(MemAccessAddress {
                    base: sbase,
                    reg_offset,
                    imm_offset: Some(imm_offset),
                })
            }
            _ => None,
        }
    }
}

impl BinOpInst {
    pub(crate) fn new(inst: Instruction) -> Self {
        Self { inst }
    }

    pub(crate) fn get_desc(&self) -> BinOpOperandMapping {
        use BinaryOperator::*;
        const DESCS: &[(Opcode, BinOpOperandMapping)] = &[
            (
                Opcode::SOPK(SOPKOpcode::S_ADDK_I32),
                BinOpOperandMapping::new(Add, 0, 0, 2),
            ),
            (
                Opcode::SOPK(SOPKOpcode::S_MULK_I32),
                BinOpOperandMapping::new(Mul, 0, 0, 1),
            ),
            (
                Opcode::SOP2(SOP2Opcode::S_ADD_U32),
                BinOpOperandMapping::new(Add, 0, 2, 3),
            ),
            (
                Opcode::SOP2(SOP2Opcode::S_ADD_I32),
                BinOpOperandMapping::new(Add, 0, 2, 3),
            ),
            (
                Opcode::SOP2(SOP2Opcode::S_SUB_U32),
                BinOpOperandMapping::new(Sub, 0, 2, 3),
            ),
            (
                Opcode::SOP2(SOP2Opcode::S_SUB_I32),
                BinOpOperandMapping::new(Sub, 0, 2, 3),
            ),
            (
                Opcode::SOP2(SOP2Opcode::S_MUL_I32),
                BinOpOperandMapping::new(Mul, 0, 1, 2),
            ),
            (
                Opcode::SOP2(SOP2Opcode::S_MIN_I32),
                BinOpOperandMapping::new(Min, 0, 2, 3),
            ),
            (
                Opcode::SOP2(SOP2Opcode::S_MIN_U32),
                BinOpOperandMapping::new(Min, 0, 2, 3),
            ),
            (
                Opcode::SOP2(SOP2Opcode::S_MAX_I32),
                BinOpOperandMapping::new(Max, 0, 2, 3),
            ),
            (
                Opcode::SOP2(SOP2Opcode::S_MAX_U32),
                BinOpOperandMapping::new(Max, 0, 2, 3),
            ),
            (
                Opcode::SOP2(SOP2Opcode::S_LSHL_B32),
                BinOpOperandMapping::new(LShift, 0, 2, 3),
            ),
            (
                Opcode::SOP2(SOP2Opcode::S_LSHR_B32),
                BinOpOperandMapping::new(AShr, 0, 2, 3),
            ),
            (
                Opcode::SOP2(SOP2Opcode::S_ASHR_I32),
                BinOpOperandMapping::new(AShr, 0, 2, 3),
            ),
            (
                Opcode::SOP2(SOP2Opcode::S_AND_B32),
                BinOpOperandMapping::new(And, 0, 2, 3),
            ),
            (
                Opcode::SOP2(SOP2Opcode::S_OR_B32),
                BinOpOperandMapping::new(Or, 0, 2, 3),
            ),
            (
                Opcode::SOP2(SOP2Opcode::S_XOR_B32),
                BinOpOperandMapping::new(Xor, 0, 2, 3),
            ),
            (
                Opcode::VOP2(VOP2Opcode::V_ADD_NC_U32),
                BinOpOperandMapping::new(Add, 0, 1, 2),
            ),
            (
                Opcode::VOP2(VOP2Opcode::V_SUB_NC_U32),
                BinOpOperandMapping::new(Sub, 0, 1, 2),
            ),
            (
                Opcode::VOP2(VOP2Opcode::V_SUBREV_NC_U32),
                BinOpOperandMapping::new(Sub, 0, 2, 1),
            ),
            (
                Opcode::VOP2(VOP2Opcode::V_LSHLREV_B32),
                BinOpOperandMapping::new(LShift, 0, 2, 1),
            ),
            (
                Opcode::VOP2(VOP2Opcode::V_LSHRREV_B32),
                BinOpOperandMapping::new(AShr, 0, 2, 1),
            ),
            (
                Opcode::VOP2(VOP2Opcode::V_ASHRREV_I32),
                BinOpOperandMapping::new(AShr, 0, 2, 1),
            ),
            (
                Opcode::VOP2(VOP2Opcode::V_AND_B32),
                BinOpOperandMapping::new(And, 0, 1, 2),
            ),
            (
                Opcode::VOP2(VOP2Opcode::V_OR_B32),
                BinOpOperandMapping::new(Or, 0, 1, 2),
            ),
            (
                Opcode::VOP2(VOP2Opcode::V_XOR_B32),
                BinOpOperandMapping::new(Or, 0, 1, 2),
            ),
            (
                Opcode::VOP2(VOP2Opcode::V_MIN_U32),
                BinOpOperandMapping::new(Min, 0, 1, 2),
            ),
            (
                Opcode::VOP2(VOP2Opcode::V_MIN_I32),
                BinOpOperandMapping::new(Min, 0, 1, 2),
            ),
            (
                Opcode::VOP2(VOP2Opcode::V_MAX_U32),
                BinOpOperandMapping::new(Max, 0, 1, 2),
            ),
            (
                Opcode::VOP2(VOP2Opcode::V_MAX_I32),
                BinOpOperandMapping::new(Max, 0, 1, 2),
            ),
            // TODO: Actually bound the operands of v_mul as 24-bit integers
            (
                Opcode::VOP2(VOP2Opcode::V_MUL_U32_U24),
                BinOpOperandMapping::new(Mul, 0, 1, 2),
            ),
            (
                Opcode::VOP2(VOP2Opcode::V_MUL_I32_I24),
                BinOpOperandMapping::new(Mul, 0, 1, 2),
            ),
            (
                Opcode::VOP3AB(VOP3ABOpcode::V_ADD_NC_U32),
                BinOpOperandMapping::new(Add, 0, 1, 2),
            ),
            (
                Opcode::VOP3AB(VOP3ABOpcode::V_ADD_NC_I32),
                BinOpOperandMapping::new(Add, 0, 1, 2),
            ),
            (
                Opcode::VOP3AB(VOP3ABOpcode::V_SUB_NC_I32),
                BinOpOperandMapping::new(Sub, 0, 1, 2),
            ),
            (
                Opcode::VOP3AB(VOP3ABOpcode::V_SUB_NC_U32),
                BinOpOperandMapping::new(Sub, 0, 1, 2),
            ),
            (
                Opcode::VOP3AB(VOP3ABOpcode::V_SUBREV_NC_U32),
                BinOpOperandMapping::new(Sub, 0, 2, 1),
            ),
            (
                Opcode::VOP3AB(VOP3ABOpcode::V_ADD_CO_U32),
                BinOpOperandMapping::new(Add, 0, 2, 3),
            ),
            (
                Opcode::VOP3AB(VOP3ABOpcode::V_SUB_CO_U32),
                BinOpOperandMapping::new(Sub, 0, 2, 3),
            ),
            (
                Opcode::VOP3AB(VOP3ABOpcode::V_MUL_LO_U32),
                BinOpOperandMapping::new(Mul, 0, 1, 2),
            ),
            (
                Opcode::VOP3AB(VOP3ABOpcode::V_LSHLREV_B32),
                BinOpOperandMapping::new(LShift, 0, 2, 1),
            ),
            (
                Opcode::VOP3AB(VOP3ABOpcode::V_LSHRREV_B32),
                BinOpOperandMapping::new(AShr, 0, 2, 1),
            ),
            (
                Opcode::VOP3AB(VOP3ABOpcode::V_ASHRREV_I32),
                BinOpOperandMapping::new(AShr, 0, 2, 1),
            ),
            (
                Opcode::VOP3AB(VOP3ABOpcode::V_AND_B32),
                BinOpOperandMapping::new(And, 0, 1, 2),
            ),
            (
                Opcode::VOP3AB(VOP3ABOpcode::V_OR_B32),
                BinOpOperandMapping::new(Or, 0, 1, 2),
            ),
            (
                Opcode::VOP3AB(VOP3ABOpcode::V_XOR_B32),
                BinOpOperandMapping::new(Or, 0, 1, 2),
            ),
        ];
        DESCS
            .iter()
            .find(|x| self.inst.op == x.0)
            .expect("Unrecognized mapping")
            .1
    }
}

impl TenaryOpInst {
    pub(crate) fn new(inst: Instruction) -> Self {
        Self { inst }
    }

    pub(crate) fn get_desc(&self) -> TernaryOperandMapping {
        use BinaryOperator::*;
        use VOP3ABOpcode::*;
        const DESCS: &[(Opcode, TernaryOperandMapping)] = &[
            (
                Opcode::VOP3AB(V_ADD3_U32),
                TernaryOperandMapping::new(Add, Add, 0, 1, 2, 3),
            ),
            (
                Opcode::VOP3AB(V_LSHL_ADD_U32),
                TernaryOperandMapping::new(LShift, Add, 0, 1, 2, 3),
            ),
            (
                Opcode::VOP3AB(V_ADD_LSHL_U32),
                TernaryOperandMapping::new(Add, LShift, 0, 1, 2, 3),
            ),
            (
                Opcode::VOP3AB(V_LSHL_OR_B32),
                TernaryOperandMapping::new(LShift, Or, 0, 1, 2, 3),
            ),
            (
                Opcode::VOP3AB(V_AND_OR_B32),
                TernaryOperandMapping::new(And, Or, 0, 1, 2, 3),
            ),
            // TODO: Actually bound the operands of v_mad as 24-bit / 16-bit integers
            (
                Opcode::VOP3AB(V_MAD_U32_U24),
                TernaryOperandMapping::new(Mul, Add, 0, 1, 2, 3),
            ),
            (
                Opcode::VOP3AB(V_MAD_U32_U16),
                TernaryOperandMapping::new(Mul, Add, 0, 1, 2, 3),
            ),
            (
                Opcode::VOP3AB(V_MAD_I32_I24),
                TernaryOperandMapping::new(Mul, Add, 0, 1, 2, 3),
            ),
            (
                Opcode::VOP3AB(V_MAD_I32_I16),
                TernaryOperandMapping::new(Mul, Add, 0, 1, 2, 3),
            ),
            // mad_64_32 are sometimes used as mad_32_32, the high-32-bits of the operands are simply ignored
            // Value::Instruction(mad64_inst_idx, 0) denotes the low-32-bits of the output,
            // while Value::Instruction(mad64_inst_idx, 1) denotes the high-32-bits.
            // Only allow access to low-32-bits by setting dst_idx to 0.
            (
                Opcode::VOP3AB(V_MAD_U64_U32),
                TernaryOperandMapping::new(Mul, Add, 0, 2, 3, 4),
            ),
            (
                Opcode::VOP3AB(V_MAD_I64_I32),
                TernaryOperandMapping::new(Mul, Add, 0, 2, 3, 4),
            ),
        ];
        DESCS
            .iter()
            .find(|x| self.inst.op == x.0)
            .expect("Unrecognized mapping")
            .1
    }
}

impl SelectInst {
    pub(crate) fn new(inst: Instruction) -> Self {
        Self { inst }
    }

    pub(crate) fn get_desc(&self) -> CSelectOperandMapping {
        const DESCS: &[(Opcode, CSelectOperandMapping)] = &[
            (
                Opcode::SOP2(SOP2Opcode::S_CSELECT_B32),
                CSelectOperandMapping::new(0, 1, 2, 3),
            ),
            (
                Opcode::VOP2(VOP2Opcode::V_CNDMASK_B32),
                CSelectOperandMapping::new(0, 3, 1, 2),
            ),
            (
                Opcode::VOP3AB(VOP3ABOpcode::V_CNDMASK_B32),
                CSelectOperandMapping::new(0, 3, 1, 2),
            ),
        ];
        DESCS
            .iter()
            .find(|x| self.inst.op == x.0)
            .expect("Unrecognized mapping")
            .1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::DomFrontier;
    use crate::fileformat::Disassembler;
    use crate::ir::{APConstant, DomTree, Value};
    use crate::isa::rdna2::Decoder;
    use crate::tests::cfg::simple_kernel_info;

    #[test]
    fn test_cmp_type() {
        const INSTRUCTIONS: &[&[u32]] = &[
            // VOPC
            &[0x7C020080], // v_cmp_lt_f32_e32 vcc_lo, 0, v0
            &[0x7C420080], // v_cmp_lt_f64_e32 vcc_lo, 0, v[0:1]
            &[0x7D020080], // v_cmp_lt_i32_e32 vcc_lo, 0, v0
            &[0x7D420080], // v_cmp_lt_i64_e32 vcc_lo, 0, v[0:1]
            // VOP3AB
            &[0xD401006A, 0x00020080], // v_cmp_lt_f32_e64 vcc_lo, 0, v0
            &[0xD421006A, 0x00020080], // v_cmp_lt_f64_e64 vcc_lo, 0, v[0:1]
            &[0xD481006A, 0x00020080], // v_cmp_lt_i32_e64 vcc_lo, 0, v0
            &[0xD4A1006A, 0x00020080], // v_cmp_lt_i64_e64 vcc_lo, 0, v[0:1]
            // SOPC
            &[0xBF008000], // s_cmp_eq_i32 s0, 0
            // SOPK
            &[0xB5000000], // s_cmpk_lg_u32 s0, 0x0
        ];
        const TYPES: &[Type] = &[
            Type::Float32,
            Type::Float64,
            Type::Int32,
            Type::Int64,
            Type::Float32,
            Type::Float64,
            Type::Int32,
            Type::Int64,
            Type::Int32,
            Type::Int32,
        ];
        INSTRUCTIONS.iter().zip(TYPES).for_each(|(inst, ty)| {
            let (_, inst) = Decoder::new(inst)
                .next()
                .expect("Can not decode instruction");
            if let Some(IRInstruction::CmpInst(cmp_inst)) = inst.wrap() {
                assert_eq!(cmp_inst.get_type(), *ty);
            } else {
                panic!()
            }
        });
    }

    #[test]
    fn test_sload() {
        // s_load_dword s0, s[4:5], 0x4
        // s_endpgm
        const CODE: &[u32] = &[0xF4000002, 0xFA000004, 0xBF810000];
        let ki = &simple_kernel_info("", CODE);
        let func = &Disassembler::parse_kernel(ki).expect("Failed to parse the function");
        let dom = DomTree::analyze(&func);
        let df = DomFrontier::new(&dom);
        let def_use = PHIAnalysis::analyze(&func, &dom, &df, &ki).expect("Cannot analyze Phi");
        let inst = func.instructions[0].clone();
        let ir_inst = inst.wrap().expect("Can not wrap into IRInstruction");
        if let IRInstruction::LoadInst(load_inst) = ir_inst {
            let mem_access = load_inst.get_addr(0, &def_use).expect("Can not get addr");
            assert!(mem_access.reg_offset.is_none());
            assert!(matches!(
                mem_access.imm_offset,
                Some(VirtualUse::Value(Value::Constant(APConstant::ConstantInt(
                    4
                ))))
            ));
        } else {
            assert!(false);
        }
    }
}
