use crate::isa::rdna2::decoder::{
    DSInstSideEffect, MUBUFInstructionType, SMEMInstExpectedOperands, SOP1InstExpectedOperands,
    SOP1InstSideEffect, SOP2InstSideEffect, SOPPInstExpectedOperands, VMemInstructionType,
    VOP2InstExpectedOperands, VOP2InstSideEffect, VOP3ABDstType,
};
use crate::isa::rdna2::isa::Opcode;
use crate::isa::rdna2::opcodes::{
    LDSGDSOpcode, MUBUFOpcode, SMEMOpcode, SOP1Opcode, SOP2Opcode, SOPKOpcode, SOPPOpcode,
    VMEMOpcode, VOP1Opcode, VOP2Opcode, VOP3ABOpcode, VOP3POpcode,
};

/**
 * Describe the effects of each operands for the instructions.
 *
 * TODO: Move some of the descriptions of the instructions in decoder here, and refactor it into the form of td.
 **/
#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub enum Effect {
    // Either a constant or void
    Noop,
    ReadOnly,
    Write,
    ReadWrite,
}

impl Effect {
    const SOPC_EFFECTS: &'static [Effect] = &[Effect::Write, Effect::ReadOnly, Effect::ReadOnly];
    const VOPC_EFFECTS: &'static [Effect] = &[Effect::Write, Effect::ReadOnly, Effect::ReadOnly];

    /// Return effects on each operand
    /// - Vector ALU / Mem instructions reads EXEC implicitly, which is not shown in the result
    /// - VGPR / SGPR indexing instructions can read or write register depends on runtime value of M0,
    ///     `get_effect` is not implemented on these instructions
    pub fn get_effect(opcode: Opcode) -> &'static [Effect] {
        match opcode {
            Opcode::SOP2(x) => Self::sop2_effect(x),
            Opcode::SOP1(x) => Self::sop1_effect(x),
            Opcode::SOPC(_) => Self::SOPC_EFFECTS,
            Opcode::SOPP(x) => Self::sopp_effect(x),
            Opcode::SOPK(x) => Self::sopk_effect(x),
            Opcode::SMEM(x) => Self::smem_effect(x),
            Opcode::VOP1(x) => Self::vop1_effect(x),
            Opcode::VOP2(x) => Self::vop2_effect(x),
            Opcode::VOPC(_) => Self::VOPC_EFFECTS,
            Opcode::VOP3P(x) => Self::vop3p_effect(x),
            Opcode::VOP3AB(x) => Self::vop3ab_effect(x),
            Opcode::VMEM(x) => Self::vmem_effect(x),
            Opcode::LDSGDS(x) => Self::ds_effect(x),
            Opcode::MUBUF(x) => Self::mubuf_effect(x),
            Opcode::INVALID(_) => unreachable!(),
            _ => todo!(),
        }
    }

    fn sop2_effect(o: SOP2Opcode) -> &'static [Effect] {
        use Effect::*;
        match SOP2InstSideEffect::from_opcode(o) {
            SOP2InstSideEffect::NoSideEffect => &[Write, ReadOnly, ReadOnly],
            SOP2InstSideEffect::WriteScc => &[Write, Write, ReadOnly, ReadOnly],
            SOP2InstSideEffect::ReadWriteScc => &[Write, ReadWrite, ReadOnly, ReadOnly],
            SOP2InstSideEffect::ReadScc => &[Write, ReadOnly, ReadOnly, ReadOnly],
        }
    }

    fn sop1_effect(o: SOP1Opcode) -> &'static [Effect] {
        use Effect::*;
        let expected_operands = SOP1InstExpectedOperands::from_opcode(o);
        let side_effect = SOP1InstSideEffect::from_opcode(o);
        use SOP1InstExpectedOperands::*;
        use SOP1InstSideEffect::*;
        match (side_effect, expected_operands) {
            (NoSideEffect, AccessDstSrc) => &[Write, ReadOnly],
            (NoSideEffect, NoSrc) => &[Write],
            (NoSideEffect, NoDst) => &[Noop, ReadOnly],
            (ReadScc, AccessDstSrc) => &[Write, ReadOnly],
            (WriteScc, AccessDstSrc) => &[Write, Write, ReadOnly],
            (ReadM0, AccessDstSrc) => {
                // these instructions indexes the dst / src by M0
                unimplemented!("SGPR indexing instructions not supported yet")
            }
            (WriteSccAndExec, AccessDstSrc) => &[Write, Write, Write, ReadOnly],
            _ => unreachable!(),
        }
    }

    fn sopk_effect(o: SOPKOpcode) -> &'static [Effect] {
        use Effect::*;
        use SOPKOpcode::*;
        match o {
            S_VERSION => &[Noop, Noop],
            S_CMOVK_I32 => &[Write, Noop, ReadOnly],
            S_WAITCNT_EXPCNT | S_WAITCNT_LGKMCNT | S_WAITCNT_VMCNT | S_WAITCNT_VSCNT => {
                &[Noop, ReadOnly, Noop]
            }
            S_CALL_B64 | S_MOVK_I32 | S_GETREG_B32 => &[Write, Noop],
            S_MULK_I32 => &[ReadWrite, Noop],
            S_ADDK_I32 => &[ReadWrite, Write, Noop],
            S_SETREG_IMM32_B32 => &[Noop, Noop, Noop],
            S_SETREG_B32 => &[Noop, Noop, ReadOnly],
            S_SUBVECTOR_LOOP_BEGIN | S_SUBVECTOR_LOOP_END => unimplemented!(
                "This opcode effects 64-bit EXEC has no practical wave32 programming scenario according to ISA"
            ),
            _ => &[Write, ReadOnly, Noop],
        }
    }

    fn sopp_effect(o: SOPPOpcode) -> &'static [Effect] {
        use Effect::*;
        match SOPPInstExpectedOperands::from_opcode(o) {
            SOPPInstExpectedOperands::ReadSimm => &[Noop, Noop],
            SOPPInstExpectedOperands::NoOperand => &[Noop],
            SOPPInstExpectedOperands::ConditionalBranch(_) => &[Noop, ReadOnly, Noop],
        }
    }

    fn smem_effect(o: SMEMOpcode) -> &'static [Effect] {
        use Effect::*;
        use SMEMInstExpectedOperands::*;
        match o.expected_operands() {
            NoOperands => &[Noop],
            NoSrc => &[Write],
            NoDst => &[Noop, Noop, ReadOnly, ReadOnly, Noop],
            AccessDstSrc => &[Write, ReadOnly, Noop, Noop, ReadOnly, Noop],
        }
    }

    fn vop1_effect(o: VOP1Opcode) -> &'static [Effect] {
        use Effect::*;
        use VOP1Opcode::*;
        match o {
            V_CLREXCP | V_PIPEFLUSH | V_NOP => &[Noop],
            V_MOVRELD_B32 | V_MOVRELS_B32 | V_MOVRELSD_B32 | V_MOVRELSD_2_B32 | V_SWAPREL_B32 => {
                // these instructions uses a relative destination / source address
                unimplemented!("instruction with SGPR indexing not supported yet")
            }
            _ => &[Write, ReadOnly],
        }
    }

    fn vop2_effect(o: VOP2Opcode) -> &'static [Effect] {
        use Effect::*;
        use VOP2InstExpectedOperands::*;
        use VOP2InstSideEffect::*;
        let side_effect = VOP2InstSideEffect::from_opcode(o);
        let expected_operands = VOP2InstExpectedOperands::from_opcode(o);
        match (side_effect, expected_operands) {
            (NoSideEffect, TwoSrcs) => &[Write, ReadOnly, ReadOnly],
            (NoSideEffect, ExtraSimmAsSrc1) => &[Write, ReadOnly, Noop, ReadOnly],
            (NoSideEffect, ExtraSimmAsSrc2) => &[Write, ReadOnly, ReadOnly, Noop],
            (ReadVcc, TwoSrcs) => &[Write, ReadOnly, ReadOnly, ReadOnly],
            (ReadAndWriteVcc, TwoSrcs) => &[Write, Write, ReadOnly, ReadOnly, ReadOnly],
            _ => unreachable!("No such case in ISA"),
        }
    }

    fn vop3p_effect(o: VOP3POpcode) -> &'static [Effect] {
        use Effect::*;
        &[Write, ReadOnly, ReadOnly, ReadOnly][0..o.src_count() + 1]
    }

    fn vop3ab_effect(o: VOP3ABOpcode) -> &'static [Effect] {
        use Effect::*;
        match VOP3ABDstType::from_opcode(o) {
            VOP3ABDstType::NoDst => {
                &[Noop, ReadOnly, ReadOnly, ReadOnly, ReadOnly][0..o.src_count() + 1]
            }
            VOP3ABDstType::VDstSDst => {
                &[Write, Write, ReadOnly, ReadOnly, ReadOnly, ReadOnly][0..o.src_count() + 2]
            }
            _ => &[Write, ReadOnly, ReadOnly, ReadOnly, ReadOnly][0..o.src_count() + 1],
        }
    }

    fn vmem_effect(o: VMEMOpcode) -> &'static [Effect] {
        use Effect::*;
        use VMEMOpcode::*;
        use VMemInstructionType::*;
        match o {
            GLOBAL_LOAD_DWORD_ADDTID => &[Write, Noop, Noop, ReadOnly, Noop, Noop],
            GLOBAL_STORE_DWORD_ADDTID => &[Noop, Noop, ReadOnly, ReadOnly, Noop, Noop],
            _ => match VMemInstructionType::from_opcode(o) {
                Load => &[Write, ReadOnly, Noop, ReadOnly, Noop, Noop],
                Store => &[Noop, ReadOnly, ReadOnly, ReadOnly, Noop, Noop],
                Atomic => &[Write, ReadOnly, ReadOnly, ReadOnly, Noop, Noop],
            },
        }
    }

    fn ds_effect(o: LDSGDSOpcode) -> &'static [Effect] {
        use Effect::*;
        let (dst_count, addr_count, data_count) = o.expected_operands();
        let side_effect_count = match o.side_effect() {
            DSInstSideEffect::NoSideEffect => 0,
            DSInstSideEffect::ReadM0 => 1,
        };
        let readonly_count = (addr_count + data_count + side_effect_count) as usize;
        let op_count = 1 + readonly_count;
        let effect: &'static [Effect] = match dst_count {
            0 => &[Noop, ReadOnly, ReadOnly, ReadOnly],
            _ => &[Write, ReadOnly, ReadOnly, ReadOnly],
        };
        &effect[..op_count]
    }

    fn mubuf_effect(o: MUBUFOpcode) -> &'static [Effect] {
        use Effect::*;
        use MUBUFInstructionType::*;
        match MUBUFInstructionType::from_opcode(o) {
            BufferAtomic => &[ReadWrite, ReadOnly, ReadOnly, ReadOnly, Noop, Noop, Noop],
            BufferLoad => &[Write, ReadOnly, ReadOnly, ReadOnly, Noop, Noop, Noop],
            BufferStore => &[
                Noop, ReadOnly, ReadOnly, ReadOnly, ReadOnly, Noop, Noop, Noop,
            ],
            BufferGL => &[Noop],
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::isa::rdna2::{
        isa::Operand,
        opcodes::{SOPCOpcode, VOPCOpcode},
        Decoder,
    };

    use super::*;
    use Effect::*;

    fn test_effects(opcodes: &[Opcode], expected_effects: &[&[Effect]]) {
        std::iter::zip(opcodes.iter(), expected_effects.iter()).for_each(|(opcode, effect)| {
            assert_eq!(&Effect::get_effect(*opcode), effect);
        })
    }

    #[test]
    fn test_ds_effects() {
        use LDSGDSOpcode::*;
        const OPCODES: [LDSGDSOpcode; 10] = [
            DS_ADD_F32,
            DS_ADD_RTN_F32,
            DS_APPEND, // read M0
            DS_CMPST_RTN_B32,
            DS_GWS_INIT,   // read M0
            DS_GWS_SEMA_P, // read M0
            DS_MSKOR_B64,
            DS_READ2ST64_B64,
            DS_WRITE2_B32,
            DS_WRXCHG2ST64_RTN_B32,
        ];
        const EFFECTS: [&[Effect]; 10] = [
            &[Noop, ReadOnly, ReadOnly],
            &[Write, ReadOnly, ReadOnly],
            &[Write, ReadOnly],
            &[Write, ReadOnly, ReadOnly, ReadOnly],
            &[Noop, ReadOnly, ReadOnly],
            &[Noop, ReadOnly],
            &[Noop, ReadOnly, ReadOnly, ReadOnly],
            &[Write, ReadOnly],
            &[Noop, ReadOnly, ReadOnly, ReadOnly],
            &[Write, ReadOnly, ReadOnly, ReadOnly],
        ];
        test_effects(&OPCODES.map(|x| Opcode::LDSGDS(x)), &EFFECTS);
    }

    #[test]
    fn test_vmem_effects() {
        use VMEMOpcode::*;
        const OPCODES: [VMEMOpcode; 6] = [
            GLOBAL_ATOMIC_ADD,
            GLOBAL_ATOMIC_SWAP_X2,
            GLOBAL_LOAD_DWORDX3,
            GLOBAL_LOAD_DWORD_ADDTID,
            GLOBAL_STORE_BYTE,
            GLOBAL_STORE_DWORD_ADDTID,
        ];
        const EFFECTS: [&[Effect]; 6] = [
            &[Write, ReadOnly, ReadOnly, ReadOnly, Noop, Noop],
            &[Write, ReadOnly, ReadOnly, ReadOnly, Noop, Noop],
            &[Write, ReadOnly, Noop, ReadOnly, Noop, Noop],
            &[Write, Noop, Noop, ReadOnly, Noop, Noop],
            &[Noop, ReadOnly, ReadOnly, ReadOnly, Noop, Noop],
            &[Noop, Noop, ReadOnly, ReadOnly, Noop, Noop],
        ];
        test_effects(&OPCODES.map(|x| Opcode::VMEM(x)), &EFFECTS);
    }

    #[test]
    fn test_mubuf_effects() {
        use MUBUFOpcode::*;
        const OPCODES: [MUBUFOpcode; 4] = [
            BUFFER_ATOMIC_ADD,
            BUFFER_GL0_INV,
            BUFFER_LOAD_DWORDX2,
            BUFFER_STORE_BYTE_D16_HI,
        ];
        const EFFECTS: [&[Effect]; 4] = [
            &[ReadWrite, ReadOnly, ReadOnly, ReadOnly, Noop, Noop, Noop],
            &[Noop],
            &[Write, ReadOnly, ReadOnly, ReadOnly, Noop, Noop, Noop],
            &[
                Noop, ReadOnly, ReadOnly, ReadOnly, ReadOnly, Noop, Noop, Noop,
            ],
        ];
        test_effects(&OPCODES.map(|x| Opcode::MUBUF(x)), &EFFECTS);
    }

    #[test]
    fn test_smem_effects() {
        use SMEMOpcode::*;
        const OPCODES: [SMEMOpcode; 4] = [S_LOAD_DWORD, S_BUFFER_LOAD_DWORD, S_GL1_INV, S_MEMTIME];
        const EFFECTS: [&[Effect]; 4] = [
            &[Write, ReadOnly, Noop, Noop, ReadOnly, Noop],
            &[Write, ReadOnly, Noop, Noop, ReadOnly, Noop],
            &[Noop],
            &[Write],
        ];
        test_effects(&OPCODES.map(|x| Opcode::SMEM(x)), &EFFECTS);
    }

    #[test]
    fn test_sopx_effects() {
        use Opcode::*;
        use SOP1Opcode::*;
        use SOP2Opcode::*;
        use SOPCOpcode::*;
        use SOPKOpcode::*;
        use SOPPOpcode::*;
        const OPCODES: [Opcode; 12] = [
            SOP1(S_ABS_I32),
            SOP1(S_GETPC_B64),
            SOP1(S_SETPC_B64),
            SOP2(S_MUL_I32),
            SOP2(S_ADD_I32),
            SOP2(S_ADDC_U32),
            SOPK(S_MULK_I32),
            SOPK(S_CMPK_EQ_I32),
            SOPC(S_BITCMP0_B32),
            SOPP(S_BARRIER),
            SOPP(S_BRANCH),
            SOPP(S_CBRANCH_SCC0),
        ];
        const EFFECTS: [&[Effect]; 12] = [
            &[Write, Write, ReadOnly],
            &[Write],
            &[Noop, ReadOnly],
            &[Write, ReadOnly, ReadOnly],
            &[Write, Write, ReadOnly, ReadOnly],
            &[Write, ReadWrite, ReadOnly, ReadOnly],
            &[ReadWrite, Noop],
            &[Write, ReadOnly, Noop],
            &[Write, ReadOnly, ReadOnly],
            &[Noop],
            &[Noop, Noop],
            &[Noop, ReadOnly, Noop],
        ];
        test_effects(&OPCODES, &EFFECTS);
    }

    #[test]
    fn test_vopx_effects() {
        use Opcode::*;
        use VOP1Opcode::*;
        use VOP2Opcode::*;
        use VOP3POpcode::*;
        use VOPCOpcode::*;
        const OPCODES: [Opcode; 12] = [
            VOP1(V_CLREXCP),
            VOP1(V_CVT_F16_F32),
            VOP2(V_ADD_CO_CI_U32),
            VOP2(V_MUL_F16),
            VOP2(V_FMAAK_F16),
            VOP3AB(VOP3ABOpcode::V_ADD3_U32),
            VOP3AB(VOP3ABOpcode::V_ADD_CO_U32),
            VOP3AB(VOP3ABOpcode::V_COS_F16),
            VOP3AB(VOP3ABOpcode::V_MUL_F16),
            VOP3P(V_DOT2_F32_F16),
            VOP3P(V_PK_ADD_F16),
            VOPC(V_CMPX_EQ_I16),
        ];
        const EFFECTS: [&[Effect]; 12] = [
            &[Noop],
            &[Write, ReadOnly],
            &[Write, Write, ReadOnly, ReadOnly, ReadOnly],
            &[Write, ReadOnly, ReadOnly],
            &[Write, ReadOnly, ReadOnly, Noop],
            &[Write, ReadOnly, ReadOnly, ReadOnly],
            &[Write, Write, ReadOnly, ReadOnly],
            &[Write, ReadOnly],
            &[Write, ReadOnly, ReadOnly],
            &[Write, ReadOnly, ReadOnly, ReadOnly],
            &[Write, ReadOnly, ReadOnly],
            &[Write, ReadOnly, ReadOnly],
        ];
        test_effects(&OPCODES, &EFFECTS);
    }

    /// assert instructions have read / write side effects on the speical register
    fn test_side_effects(instructions: &[&[u32]], special_reg: u8, write: bool) {
        instructions.iter().for_each(|binary| {
            let (_, inst) = Decoder::new(binary)
                .next()
                .expect("can not decode instruction");
            let effects = Effect::get_effect(inst.op);
            assert!(
                std::iter::zip(effects.iter(), inst.get_operands().iter()).any(
                    |(effect, operand)| match operand {
                        Operand::SpecialScalarRegister(r) if *r == special_reg => {
                            matches!(
                                (effect, write),
                                (Write, true) | (ReadOnly, false) | (ReadWrite, _)
                            )
                        }
                        _ => false,
                    }
                )
            );
        })
    }

    #[test]
    fn test_read_scc() {
        const INSTRUCTIONS: [&[u32]; 6] = [
            &[0xBF840000], // s_cbranch_scc0 0
            &[0x85008180], // s_cselect_b32 s0, 0, 1
            &[0xBE800501], // s_cmov_b32 s0, s1
            &[0xB1000000], // s_cmovk_i32 s0, 0x0
            &[0x82000201], // s_addc_u32 s0, s1, s2
            &[0x82800201], // s_subb_u32 s0, s1, s2
        ];
        test_side_effects(&INSTRUCTIONS, Operand::SPECIAL_REG_SCC, false);
    }

    #[test]
    fn test_write_scc() {
        const INSTRUCTIONS: [&[u32]; 19] = [
            &[0x81000201], // s_add_i32 s0, s1, s2
            &[0x82000201], // s_addc_u32 s0, s1, s2
            &[0x81800201], // s_sub_i32 s0, s1, s2
            &[0x82800201], // s_subb_u32 s0, s1, s2
            &[0x96000201], // s_absdiff_i32 s0, s1, s2
            &[0x84000201], // s_max_i32 s0, s1, s2
            &[0x83000201], // s_min_i32 s0, s1, s2
            &[0xB7800000], // s_addk_i32 s0, 0x0
            &[0xBE803401], // s_abs_i32 s0, s1
            &[0xBF060100], // s_cmp_eq_u32 s0, s1
            &[0xB1800000], // s_cmpk_eq_i32 s0, 0x0
            &[0x87000201], // s_and_b32 s0, s1, s2
            &[0x8F000201], // s_lshl_b32 s0, s1, s2
            &[0x93800201], // s_bfe_u32 s0, s1, s2
            &[0xBE800701], // s_not_b32 s0, s1
            &[0xBE800901], // s_wqm_b32 s0, s1
            &[0xBE800D01], // s_bcnt0_i32_b32 s0, s1
            &[0xBE803C01], // s_and_saveexec_b32 s0, s1
            &[0xBE804601], // s_andn1_wrexec_b32 s0, s1
        ];
        test_side_effects(&INSTRUCTIONS, Operand::SPECIAL_REG_SCC, true);
    }

    #[test]
    fn test_read_vcc() {
        const INSTRUCTIONS: [&[u32]; 2] = [
            &[0x50000501], // v_add_co_ci_u32_e32 v0, vcc_lo, v1, v2, vcc_lo
            &[0xBF860000], // s_cbranch_vccz 0
        ];
        test_side_effects(&INSTRUCTIONS, Operand::SPECIAL_REG_VCC_LO, false);
    }

    #[test]
    fn test_write_vcc() {
        const INSTRUCTIONS: [&[u32]; 5] = [
            &[0x7D1E0300],             // v_cmp_class_f16_e32 vcc_lo, v0, v1
            &[0x7C080300],             // v_cmp_gt_f32_e32 vcc_lo, v0, v1
            &[0x50000501],             // v_add_co_ci_u32_e32 v0, vcc_lo, v1, v2, vcc_lo
            &[0xD70F6A00, 0x00020501], // v_add_co_u32 v0, vcc_lo, v1, v2
            &[0xD56D6A00, 0x040E0501], // v_div_scale_f32 v0, vcc_lo, v1, v2, v3
        ];
        test_side_effects(&INSTRUCTIONS, Operand::SPECIAL_REG_VCC_LO, true);
    }

    #[test]
    fn test_write_exec() {
        const INSTRUCTIONS: [&[u32]; 3] = [
            &[0xBE803C01], // s_and_saveexec_b32 s0, s1
            &[0x7D3E0300], // v_cmpx_class_f16_e32 v0, v1
            &[0x7C280300], // v_cmpx_gt_f32_e32 v0, v1
        ];
        test_side_effects(&INSTRUCTIONS, Operand::SPECIAL_REG_EXEC_LO, true);
    }
}
