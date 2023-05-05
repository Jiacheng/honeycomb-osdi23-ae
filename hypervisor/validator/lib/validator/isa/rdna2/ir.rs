use crate::ir::instruction::{
    AtomicInst, BinOpInst, BranchInst, CmpInst, IRInstruction, LoadInst, SelectInst, StoreInst,
    TenaryOpInst,
};
use crate::isa::rdna2::decoder::VMemInstructionType;
use crate::isa::rdna2::isa::Opcode;
use crate::isa::rdna2::opcodes::{SMEMOpcode, SOP2Opcode, SOPKOpcode, VOP2Opcode, VOP3ABOpcode};
use crate::isa::rdna2::Instruction;
use crate::isa::rdna2::VOP3ABInstType;

impl Instruction {
    pub fn wrap(self) -> Option<IRInstruction> {
        match self.op {
            Opcode::SOPP(opcode) if opcode.is_branch() => {
                Some(IRInstruction::BranchInst(BranchInst::new(self)))
            }
            Opcode::SOPC(_) | Opcode::VOPC(_) => Some(IRInstruction::CmpInst(CmpInst::new(self))),
            Opcode::VOP3AB(opcode)
                if matches!(
                    VOP3ABInstType::from_opcode(opcode),
                    VOP3ABInstType::FromVOPC
                ) =>
            {
                Some(IRInstruction::CmpInst(CmpInst::new(self)))
            }
            Opcode::SOPK(opcode) => {
                use SOPKOpcode::*;
                match opcode {
                    S_CMPK_EQ_I32 | S_CMPK_LG_I32 | S_CMPK_GT_I32 | S_CMPK_GE_I32
                    | S_CMPK_LT_I32 | S_CMPK_LE_I32 | S_CMPK_EQ_U32 | S_CMPK_LG_U32
                    | S_CMPK_GT_U32 | S_CMPK_GE_U32 | S_CMPK_LT_U32 | S_CMPK_LE_U32 => {
                        Some(IRInstruction::CmpInst(CmpInst::new(self)))
                    }
                    S_ADDK_I32 | S_MULK_I32 => Some(IRInstruction::BinOpInst(BinOpInst::new(self))),
                    _ => None,
                }
            }
            Opcode::SOP2(opcode) => {
                use SOP2Opcode::*;
                match opcode {
                    S_ADD_U32 | S_ADD_I32 | S_SUB_U32 | S_SUB_I32 | S_MUL_I32 | S_LSHL_B32
                    | S_LSHR_B32 | S_ASHR_I32 | S_AND_B32 | S_OR_B32 | S_XOR_B32 | S_MIN_I32
                    | S_MIN_U32 | S_MAX_I32 | S_MAX_U32 => {
                        Some(IRInstruction::BinOpInst(BinOpInst::new(self)))
                    }
                    S_CSELECT_B32 => Some(IRInstruction::SelectInst(SelectInst::new(self))),
                    _ => None,
                }
            }
            Opcode::VOP2(opcode) => {
                use VOP2Opcode::*;
                match opcode {
                    V_ADD_NC_U32 | V_SUB_NC_U32 | V_SUBREV_NC_U32 | V_LSHLREV_B32
                    | V_LSHRREV_B32 | V_ASHRREV_I32 | V_AND_B32 | V_OR_B32 | V_XOR_B32
                    | V_MIN_I32 | V_MAX_I32 | V_MIN_U32 | V_MAX_U32 | V_MUL_U32_U24
                    | V_MUL_I32_I24 => Some(IRInstruction::BinOpInst(BinOpInst::new(self))),
                    V_CNDMASK_B32 => Some(IRInstruction::SelectInst(SelectInst::new(self))),
                    _ => None,
                }
            }
            Opcode::VOP3AB(opcode) => {
                use VOP3ABOpcode::*;
                match opcode {
                    V_ADD_NC_U32 | V_ADD_NC_I32 | V_SUB_NC_U32 | V_SUBREV_NC_U32 | V_SUB_NC_I32
                    | V_ADD_CO_U32 | V_SUB_CO_U32 | V_MUL_LO_U32 | V_LSHLREV_B32 | V_XOR_B32
                    | V_LSHRREV_B32 | V_ASHRREV_I32 | V_MIN_I32 | V_MAX_I32 | V_MIN_U32
                    | V_MAX_U32 => Some(IRInstruction::BinOpInst(BinOpInst::new(self))),
                    V_ADD3_U32 | V_LSHL_ADD_U32 | V_ADD_LSHL_U32 | V_LSHL_OR_B32 | V_AND_OR_B32
                    | V_MAD_U32_U24 | V_MAD_U32_U16 | V_MAD_I32_I24 | V_MAD_I32_I16
                    | V_MAD_U64_U32 | V_MAD_I64_I32 => {
                        Some(IRInstruction::TernaryOpInst(TenaryOpInst::new(self)))
                    }
                    V_CNDMASK_B32 => Some(IRInstruction::SelectInst(SelectInst::new(self))),
                    _ => None,
                }
            }
            Opcode::VMEM(opcode) => {
                use VMemInstructionType::*;
                match VMemInstructionType::from_opcode(opcode) {
                    Load => Some(IRInstruction::LoadInst(LoadInst::new(self))),
                    Store => Some(IRInstruction::StoreInst(StoreInst::new(self))),
                    Atomic => Some(IRInstruction::AtomicInst(AtomicInst::new(self))),
                }
            }
            Opcode::SMEM(opcode) => {
                use SMEMOpcode::*;
                match opcode {
                    S_LOAD_DWORD | S_LOAD_DWORDX2 | S_LOAD_DWORDX4 | S_LOAD_DWORDX8
                    | S_LOAD_DWORDX16 => Some(IRInstruction::LoadInst(LoadInst::new(self))),
                    _ => None,
                }
            }
            _ => None,
        }
    }
}
