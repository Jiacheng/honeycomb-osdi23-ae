use crate::analysis::{InstructionUse, PHIAnalysis};
use crate::ir::{APConstant, Value};
use crate::isa::rdna2::isa::Opcode;
use crate::isa::rdna2::opcodes::{SOP2Opcode, VOP1Opcode, VOP2Opcode, VOP3ABOpcode, VOPCOpcode};

// integer division pattern from
// https://github.com/ROCm-Developer-Tools/amd-llvm-project/blob/amd-stg-open/llvm/lib/Target/AMDGPU/AMDGPUCodeGenPrepare.cpp#L1097
// based on ideas from "Software Integer Division", Tom Rodeheffer, August 2008.
//
// and constant integer division pattern from
// https://github.com/ROCm-Developer-Tools/amd-llvm-project/blob/11d5fa11d52cc4beca16e57b9a16a56947e58635/llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp#L22997

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum DivPattern {
    Float(Value),              // float(y)
    Neg(Value),                // -y
    Frcp(Value),               // 1.0 / float(y)
    Finv0(Value),              // (2^32 - 256) / float(y)
    Inv0(Value),               // (2^32 - 256) / y
    Inv0Mul(Value),            // -y * Inv0(Value)
    Inv0Delta(Value),          // mulhi(Inv0Mul(y), Inv0(Value))
    Inv(Value),                // Inv0 + InvDelta
    Quot0(Value, Value),       // mulhi(x, Inv(y))
    Quot0Mul(Value, Value),    // Quot0(x, y) * y
    Rem0(Value, Value),        // x - Quot0Mul(x, y)
    Quot0Inc(Value, Value),    // 1 + Quot0(x, y)
    Rem0Sub(Value, Value),     // Rem0(x, y) - y
    RefineCond0(Value, Value), // Rem0(x, y) > y
    Quot1(Value, Value),       // RefineCond0 ? Quot0Inc : Quot0
    Rem1(Value, Value),        // RefineCond0 ? Rem0 : Rem0Sub
    Quot1Inc(Value, Value),    // 1 + Quot1(x, y)
    Rem1Sub(Value, Value),     // Rem1(x, y) - y
    RefineCond1(Value, Value), // Rem1(x, y) > y
    Div(Value, Value),         // RefineCond1 ? Quot1Inc : Quot1
    Rem(Value, Value),         // RefineCond1 ? Rem1 : Rem1Sub
    // following enums are for optimized constant division
    LShr(Value, isize),               // lshr(x, offset)
    MulMagic(Value, isize),           // mulh(x, MagicNumber)
    MulMagicShr(Value, isize, isize), // mulh(x, MagicNumber) >> offset
    Sign(Value),                      // lshr(x, 31)
    DivConst(Value, isize),           // MulMagicShr(x) + Sign(x) or MulMagic(x) + Sign(x)
}

impl DivPattern {
    pub(crate) fn try_match(v: Value, du: &PHIAnalysis) -> Option<Self> {
        match v {
            Value::Instruction(inst_idx, op_idx) => Self::try_match_inst(inst_idx, op_idx, du),
            _ => None,
        }
    }

    // TODO: use IRInstructions
    fn try_match_inst(inst_idx: usize, op_idx: usize, du: &PHIAnalysis) -> Option<Self> {
        use DivPattern::*;
        use Opcode::*;
        use SOP2Opcode::*;
        use VOP1Opcode::*;
        use VOP2Opcode::*;
        use VOP3ABOpcode::{V_MUL_HI_U32, V_MUL_LO_U32};
        use VOPCOpcode::*;
        let get_op = |op_idx| du.get_def_use(InstructionUse::op(inst_idx, op_idx, 0));
        let opcode = du.func.instructions[inst_idx].op;
        match opcode {
            VOP1(V_CVT_F32_U32) | VOP1(V_CVT_F32_UBYTE0) => Some(Float(get_op(1)?)),
            VOP2(V_SUB_NC_U32) => {
                let lhs = get_op(1)?;
                let rhs = get_op(2)?;
                if lhs == Value::Constant(APConstant::ConstantInt(0)) {
                    return Some(Neg(rhs));
                }
                if let Some(Quot0Mul(x, y)) = Self::try_match(rhs, du) {
                    if x == lhs {
                        return Some(Rem0(x, y));
                    }
                }
                if let Some(Rem0(x, y)) = Self::try_match(lhs, du) {
                    if y == rhs {
                        return Some(Rem0Sub(x, y));
                    }
                }
                if let Some(Rem1(x, y)) = Self::try_match(lhs, du) {
                    if y == rhs {
                        return Some(Rem1Sub(x, y));
                    }
                }
                None
            }
            VOP2(V_SUBREV_NC_U32) => {
                let lhs = get_op(2)?;
                let rhs = get_op(1)?;
                if lhs == Value::Constant(APConstant::ConstantInt(0)) {
                    return Some(Neg(rhs));
                }
                if let Some(Quot0Mul(x, y)) = Self::try_match(rhs, du) {
                    if x == lhs {
                        return Some(Rem0(x, y));
                    }
                }
                if let Some(Rem0(x, y)) = Self::try_match(lhs, du) {
                    if y == rhs {
                        return Some(Rem0Sub(x, y));
                    }
                }
                if let Some(Rem1(x, y)) = Self::try_match(lhs, du) {
                    if y == rhs {
                        return Some(Rem1Sub(x, y));
                    }
                }
                None
            }
            SOP2(S_SUB_I32) | SOP2(S_SUB_U32) => {
                let lhs = get_op(2)?;
                let rhs = get_op(3)?;
                if lhs == Value::Constant(APConstant::ConstantInt(0)) {
                    return Some(Neg(rhs));
                }
                if let Some(Quot0Mul(x, y)) = Self::try_match(rhs, du) {
                    if x == lhs {
                        return Some(Rem0(x, y));
                    }
                }
                if let Some(Rem0(x, y)) = Self::try_match(lhs, du) {
                    if y == rhs {
                        return Some(Rem0Sub(x, y));
                    }
                }
                if let Some(Rem1(x, y)) = Self::try_match(lhs, du) {
                    if y == rhs {
                        return Some(Rem1Sub(x, y));
                    }
                }
                None
            }
            VOP1(V_RCP_IFLAG_F32) => match Self::try_match(get_op(1)?, du)? {
                Float(v) => Some(Frcp(v)),
                _ => None,
            },
            VOP2(V_MUL_F32) => {
                if get_op(1)? == Value::Constant(APConstant::ConstantInt(0x4f7ffffe)) {
                    match Self::try_match(get_op(2)?, du)? {
                        Frcp(v) => Some(Finv0(v)),
                        _ => None,
                    }
                } else {
                    None
                }
            }
            VOP1(V_CVT_U32_F32) => match Self::try_match(get_op(1)?, du)? {
                Finv0(v) => Some(Inv0(v)),
                _ => None,
            },
            VOP3AB(V_MUL_LO_U32) => {
                let (lhs, rhs) = (get_op(1)?, get_op(2)?);
                match Self::try_match(lhs, du)? {
                    Neg(y) if Self::try_match(rhs, du)? == Inv0(y) => Some(Inv0Mul(y)),
                    Quot0(x, y) if y == rhs => Some(Quot0Mul(x, y)),
                    _ => None,
                }
            }
            VOP3AB(VOP3ABOpcode::V_MAD_U64_U32) if op_idx == 1 => {
                let (lhs, rhs, adden) = (get_op(2)?, get_op(3)?, get_op(4)?);
                if adden == Value::Constant(APConstant::ConstantInt(0)) {
                    match Self::try_match(rhs, du)? {
                        Inv0(y) if Self::try_match(lhs, du)? == Inv0Mul(y) => Some(Inv0Delta(y)),
                        Inv0Mul(y) if Self::try_match(lhs, du)? == Inv0(y) => Some(Inv0Delta(y)),
                        Inv(y) => Some(Quot0(lhs, y)),
                        _ => None,
                    }
                } else {
                    None
                }
            }
            VOP3AB(V_MUL_HI_U32) => {
                let (lhs, rhs) = (get_op(1)?, get_op(2)?);
                match Self::try_match(rhs, du)? {
                    Inv0(y) if Self::try_match(lhs, du)? == Inv0Mul(y) => Some(Inv0Delta(y)),
                    Inv0Mul(y) if Self::try_match(lhs, du)? == Inv0(y) => Some(Inv0Delta(y)),
                    Inv(y) => Some(Quot0(lhs, y)),
                    _ => None,
                }
            }
            VOP2(V_ADD_NC_U32) => {
                let (lhs, rhs) = (get_op(1)?, get_op(2)?);
                if lhs == Value::Constant(APConstant::ConstantInt(1)) {
                    match Self::try_match(rhs, du)? {
                        Quot0(x, y) => Some(Quot0Inc(x, y)),
                        Quot1(x, y) => Some(Quot1Inc(x, y)),
                        _ => None,
                    }
                } else {
                    match (Self::try_match(lhs, du)?, Self::try_match(rhs, du)?) {
                        (Inv0(a), Inv0Delta(b)) if a == b => Some(Inv(a)),
                        _ => None,
                    }
                }
            }
            VOPC(V_CMP_GE_U32) | VOP3AB(VOP3ABOpcode::V_CMP_GE_U32) => {
                let lhs = get_op(1)?;
                let rhs = get_op(2)?;
                match Self::try_match(lhs, du)? {
                    Rem0(x, y) if y == rhs => Some(RefineCond0(x, y)),
                    Rem1(x, y) if y == rhs => Some(RefineCond1(x, y)),
                    _ => None,
                }
            }
            VOPC(V_CMP_LE_U32) | VOP3AB(VOP3ABOpcode::V_CMP_LE_U32) => {
                let lhs = get_op(1)?;
                let rhs = get_op(2)?;
                match Self::try_match(rhs, du)? {
                    Rem0(x, y) if y == lhs => Some(RefineCond0(x, y)),
                    Rem1(x, y) if y == lhs => Some(RefineCond1(x, y)),
                    _ => None,
                }
            }
            VOP2(V_CNDMASK_B32) | VOP3AB(VOP3ABOpcode::V_CNDMASK_B32) => {
                let cond = Self::try_match(get_op(3)?, du)?;
                let lhs = Self::try_match(get_op(1)?, du)?;
                let rhs = Self::try_match(get_op(2)?, du)?;
                match cond {
                    RefineCond0(x, y) => {
                        if rhs == Quot0Inc(x, y) && lhs == Quot0(x, y) {
                            return Some(Quot1(x, y));
                        }
                        if rhs == Rem0Sub(x, y) && lhs == Rem0(x, y) {
                            return Some(Rem1(x, y));
                        }
                    }
                    RefineCond1(x, y) => {
                        if rhs == Quot1Inc(x, y) && lhs == Quot1(x, y) {
                            return Some(Div(x, y));
                        }
                        if rhs == Rem1Sub(x, y) && lhs == Rem1(x, y) {
                            return Some(Rem(x, y));
                        }
                    }
                    _ => {}
                }
                None
            }
            SOP2(S_MUL_HI_I32) => {
                let (lhs, rhs) = (get_op(1)?, get_op(2)?);
                if let Value::Constant(APConstant::ConstantInt(magic)) = lhs {
                    return Some(MulMagic(rhs, magic));
                }
                if let Value::Constant(APConstant::ConstantInt(magic)) = rhs {
                    return Some(MulMagic(lhs, magic));
                }
                None
            }
            SOP2(S_MUL_HI_U32) => {
                let (lhs, rhs) = (get_op(1)?, get_op(2)?);
                if let Value::Constant(APConstant::ConstantInt(magic)) = lhs {
                    if let Some(LShr(_, offset)) = Self::try_match(rhs, du) {
                        return Self::maybe_divconst_pattern(
                            rhs,
                            1 << (32 - offset),
                            32,
                            magic as u32 as isize,
                        );
                    }
                    return Self::maybe_divconst_pattern(rhs, 1 << 32, 32, magic as u32 as isize);
                }
                if let Value::Constant(APConstant::ConstantInt(magic)) = rhs {
                    if let Some(LShr(_, offset)) = Self::try_match(lhs, du) {
                        return Self::maybe_divconst_pattern(
                            lhs,
                            1 << (32 - offset),
                            32,
                            magic as u32 as isize,
                        );
                    }
                    return Self::maybe_divconst_pattern(lhs, 1 << 32, 32, magic as u32 as isize);
                }
                None
            }
            SOP2(S_ADD_I32) => {
                let (lhs, rhs) = (get_op(2)?, get_op(3)?);
                if let Some(MulMagic(val, magic)) = Self::try_match(lhs, du) {
                    if val == rhs {
                        return Some(MulMagic(val, magic + (1 << 32)));
                    }
                }
                if let Some(MulMagic(val, magic)) = Self::try_match(rhs, du) {
                    if val == lhs {
                        return Some(MulMagic(val, magic + (1 << 32)));
                    }
                }
                match (Self::try_match(lhs, du)?, Self::try_match(rhs, du)?) {
                    (MulMagic(val, magic), Sign(val0)) | (Sign(val0), MulMagic(val, magic))
                        if val == val0 =>
                    {
                        Self::maybe_divconst_pattern(val, 1 << 31, 32, magic)
                    }
                    (MulMagicShr(val, magic, offset), Sign(val0))
                    | (Sign(val0), MulMagicShr(val, magic, offset))
                        if val == val0 =>
                    {
                        Self::maybe_divconst_pattern(val, 1 << 31, 32 + offset, magic)
                    }
                    _ => None,
                }
            }
            SOP2(S_ASHR_I32) => {
                let (lhs, rhs) = (get_op(2)?, get_op(3)?);
                if let (
                    Some(MulMagic(val, magic)),
                    Value::Constant(APConstant::ConstantInt(offset)),
                ) = (Self::try_match(lhs, du), rhs)
                {
                    return Some(MulMagicShr(val, magic, offset));
                }
                None
            }
            SOP2(S_LSHR_B32) => {
                let (lhs, rhs) = (get_op(2)?, get_op(3)?);
                if let Value::Constant(APConstant::ConstantInt(offset)) = rhs {
                    if offset == 31 {
                        return match Self::try_match(lhs, du) {
                            Some(MulMagic(val, _)) | Some(MulMagicShr(val, _, _)) => {
                                Some(Sign(val))
                            }
                            _ => Some(Sign(lhs)),
                        };
                    }
                    return Some(LShr(lhs, offset));
                }
                None
            }
            _ => None,
        }
    }

    // (x * magic) >> offset, where x < val_max
    fn maybe_divconst_pattern(
        val: Value,
        val_max: isize,
        offset: isize,
        magic_num: isize,
    ) -> Option<DivPattern> {
        let power = 1 << offset;
        let divisor = 1 + power / magic_num;
        if (val_max / divisor) * (divisor * magic_num - power) + (divisor - 1) * magic_num < power {
            // let x = q * divisor + r, 0 <= r < divisor
            //
            // x * magic - q * power
            // = q * divisor * magic + r * magic - q * power
            // >= q * (divisor * magic - power)
            // >= 0
            //
            // x * magic - q * power
            // = q * divisor * magic + r * magic - q * power
            // < (val_max / divisor) * (divisor * magic - power) + (divisor - 1) * magic
            // < power
            //
            // so x / divisor = q = x * magic / power
            Some(DivPattern::DivConst(val, divisor))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        analysis::{match_div::DivPattern, DomFrontier, PHIAnalysis},
        fileformat::Disassembler,
        ir::{DomTree, Value},
        tests::cfg::simple_kernel_info,
    };

    #[test]
    fn test_match_div() {
        use DivPattern::*;
        const CODE: &[u32] = &[
            0x7e020c0f, // v_cvt_f32_u32_e32 v1, s15
            0x818a0f80, // s_sub_i32 s10, 0, s15
            0x7e025701, // v_rcp_iflag_f32_e32 v1, v1
            0x100202ff, 0x4f7ffffe, // v_mul_f32_e32 v1, 0x4f7ffffe, v1
            0x7e020f01, // v_cvt_u32_f32_e32 v1, v1
            0xd5690003, 0x0002020a, // v_mul_lo_u32 v3, s10, v1
            0xd56a0003, 0x00020701, // v_mul_hi_u32 v3, v1, v3
            0x4a000701, // v_add_nc_u32_e32 v0, v1, v3
            0xd5767d00, 0x02020104, // v_mad_u64_u32 v[0:1], null, v4, v0, 0
            0xd5690000, 0x00001f01, // v_mul_lo_u32 v0, v1, s15
            0x4a060281, // v_add_nc_u32_e32 v3, 1, v1
            0x4c000104, // v_sub_nc_u32_e32 v0, v4, v0
            0x4e0a000f, // v_subrev_nc_u32_e32 v5, s15, v0
            0x7d86000f, // v_cmp_le_u32_e32 vcc_lo, s15, v0
            0x02020701, // v_cndmask_b32_e32 v1, v1, v3, vcc_lo
            0x02000b00, // v_cndmask_b32_e32 v0, v0, v5, vcc_lo
            0x4a060281, // v_add_nc_u32_e32 v3, 1, v1
            0x7d86000f, // v_cmp_le_u32_e32 vcc_lo, s15, v0
            0x02000701, // v_cndmask_b32_e32 v0, v1, v3, vcc_lo
        ];
        let ki = &simple_kernel_info("", &CODE);
        let func = Disassembler::parse_kernel(ki).expect("Failed to parse the function");
        let dom = DomTree::analyze(&func);
        let df = DomFrontier::new(&dom);
        let def_use = PHIAnalysis::analyze(&func, &dom, &df, &ki).expect("Cannot analyze Phi");
        let m = |i, j| DivPattern::try_match(Value::Instruction(i, j), &def_use).unwrap();
        assert!(matches!(m(0, 0), Float(..)));
        assert!(matches!(m(1, 0), Neg(..)));
        assert!(matches!(m(2, 0), Frcp(..)));
        assert!(matches!(m(3, 0), Finv0(..)));
        assert!(matches!(m(4, 0), Inv0(..)));
        assert!(matches!(m(5, 0), Inv0Mul(..)));
        assert!(matches!(m(6, 0), Inv0Delta(..)));
        assert!(matches!(m(7, 0), Inv(..)));
        assert!(matches!(m(8, 1), Quot0(..)));
        assert!(matches!(m(9, 0), Quot0Mul(..)));
        assert!(matches!(m(10, 0), Quot0Inc(..)));
        assert!(matches!(m(11, 0), Rem0(..)));
        assert!(matches!(m(12, 0), Rem0Sub(..)));
        assert!(matches!(m(13, 0), RefineCond0(..)));
        assert!(matches!(m(14, 0), Quot1(..)));
        assert!(matches!(m(15, 0), Rem1(..)));
        assert!(matches!(m(16, 0), Quot1Inc(..)));
        assert!(matches!(m(17, 0), RefineCond1(..)));
        assert!(matches!(m(18, 0), Div(..)));
    }

    #[test]
    fn test_match_div_const() {
        use DivPattern::*;
        const CODE: &[u32] = &[
            0x9B03FF02, 0x92492493, // s_mul_hi_i32 s3, s2, 0x92492493
            0x81030203, // s_add_i32 s3, s3, s2
            0x90029F03, // s_lshr_b32 s2, s3, 31
            0x91038503, // s_ashr_i32 s3, s3, 5
            0x81020203, // s_add_i32 s2, s3, s2
            0x90018301, // s_lshr_b32 s1, s1, 3
            0x9a80ff01, 0x24924925, // s_mul_hi_u32 s0, s1, 0x24924925
        ];
        let ki = &simple_kernel_info("", &CODE);
        let func = Disassembler::parse_kernel(ki).expect("Failed to parse the function");
        let dom = DomTree::analyze(&func);
        let df = DomFrontier::new(&dom);
        let def_use = PHIAnalysis::analyze(&func, &dom, &df, &ki).expect("Cannot analyze Phi");
        let m = |i, j| DivPattern::try_match(Value::Instruction(i, j), &def_use).unwrap();
        assert!(matches!(m(0, 0), MulMagic(..)));
        assert!(matches!(m(1, 0), MulMagic(_, 0x92492493)));
        assert!(matches!(m(2, 0), Sign(..)));
        assert!(matches!(m(3, 0), MulMagicShr(_, 0x92492493, 5)));
        assert!(matches!(m(4, 0), DivConst(_, 56)));
        assert!(matches!(m(5, 0), LShr(_, 3)));
        assert!(matches!(m(6, 0), DivConst(_, 7)));
    }
}
