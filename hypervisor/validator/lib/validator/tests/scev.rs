use crate::analysis::{
    ConstantPropagation, DomFrontier, LoopAnalysis, PHIAnalysis, ScalarEvolution,
};
use crate::fileformat::{Disassembler, KernelInfo, SGPRSetup};
use crate::ir::{DomTree, Function};
use crate::tests::cfg::simple_kernel_info;

pub(crate) fn dummy() -> (&'static KernelInfo<'static>, Function<'static>) {
    const CODE: &[u32] = &[0xbf810000];
    const KERNEL_INFO: &KernelInfo = &simple_kernel_info("", &CODE);
    let func = Disassembler::parse_kernel(KERNEL_INFO).expect("Failed to parse the function");
    (KERNEL_INFO, func)
}

pub(crate) fn test_with_dummy_scev(f: fn(&ScalarEvolution) -> ()) {
    let (ki, func) = dummy();
    let dom = DomTree::analyze(&func);
    let df = DomFrontier::new(&dom);
    let def_use = PHIAnalysis::analyze(&func, &dom, &df, &ki).expect("Cannot analyze PHI");
    let def_use = ConstantPropagation::run(def_use);
    let dom = DomTree::analyze(&func);
    let mut li = LoopAnalysis::new(&func);
    li.analyze(&dom);
    let scev = ScalarEvolution::new(&dom, &def_use, &li, None);
    f(&scev)
}

pub(crate) fn bit_operations() -> (&'static KernelInfo<'static>, Function<'static>) {
    const CODE: &[u32] = &[
        0xf4000182, 0xfa000004, // s_load_dword s6, s[4:5], 0x4
        0x90079006, // s_lshr_b32 s7, s6, 16
        0x8707ff06, 0x0000ffff, // s_and_b32 s7, s6, 0xffff
        0xd76f0000, 0x04011208, // v_lshl_or_b32 v0, s8, 9, v0
        0x2c02008c, // v_lshrrev_b32_e32 v1, 12, v0
        0x360400ff, 0x00000fff, // v_and_b32_e32 v2, 0xfff, v0
        0xbf810000, // s_endpgm
    ];
    const KERNEL_INFO: &KernelInfo = &KernelInfo {
        name: "bit_operations",
        code: CODE,
        pgmrsrcs: [1622081536, 144, 0],
        setup: SGPRSetup::from_bits_truncate(11),
        kern_arg_size: 8,
        kern_arg_segment_align: 0,
        group_segment_fixed_size: 0,
        private_segment_fixed_size: 0,
        arguments: vec![],
    };

    let func = Disassembler::parse_kernel(KERNEL_INFO).expect("Failed to parse the function");
    (KERNEL_INFO, func)
}

pub(crate) fn argument_loop_step() -> (&'static KernelInfo<'static>, Function<'static>) {
    const CODE: &[u32] = &[
        // BB0:
        0xf4000182, 0xfa00000c, // s_load_dword s6, s[4:5], 0xc
        0xbe870380, // s_mov_b32 s7, 0
        0x93089006, // s_mul_i32 s8, s6, 16
        // BB1:
        0x81070607, // s_add_i32 s7, s7, s6
        0xbf050807, // s_cmp_lt_i32 s7, s8
        0xbf85fffd, // s_cbranch_scc1 BB1
        // BB2:
        0xbf810000, // s_endpgm
    ];
    const KERNEL_INFO: &KernelInfo = &KernelInfo {
        name: "argument_loop_step",
        code: CODE,
        pgmrsrcs: [1622081536, 144, 0],
        setup: SGPRSetup::from_bits_truncate(11),
        kern_arg_size: 8,
        kern_arg_segment_align: 0,
        group_segment_fixed_size: 0,
        private_segment_fixed_size: 0,
        arguments: vec![],
    };

    let func = Disassembler::parse_kernel(KERNEL_INFO).expect("Failed to parse the function");
    (KERNEL_INFO, func)
}

pub(crate) fn abs_pattern() -> (&'static KernelInfo<'static>, Function<'static>) {
    const CODE: &[u32] = &[
        0x81888c08, // s_sub_i32 s8, s8, 12
        0x91009f08, // s_ashr_i32 s0, s8, 31
        0x81010800, // s_add_i32 s1, s0, s8
        0x89000100, // s_xor_b32 s0, s0, s1
        0xbf810000, // s_endpgm
    ];
    const KERNEL_INFO: &KernelInfo = &KernelInfo {
        name: "abs_pattern",
        code: CODE,
        pgmrsrcs: [1622081536, 144, 0],
        setup: SGPRSetup::from_bits_truncate(11),
        kern_arg_size: 8,
        kern_arg_segment_align: 0,
        group_segment_fixed_size: 0,
        private_segment_fixed_size: 0,
        arguments: vec![],
    };

    let func = Disassembler::parse_kernel(KERNEL_INFO).expect("Failed to parse the function");
    (KERNEL_INFO, func)
}

pub(crate) fn nested_loop() -> (&'static KernelInfo<'static>, Function<'static>) {
    const CODE: &[u32] = &[
        // BB0:
        0xf4040002, 0xfa000000, // s_load_dwordx2 s[0:1], s[4:5], 0x0
        0x7e000280, // v_mov_b32_e32 v0, 0
        0x7e0202aa, // v_mov_b32_e32 v1, 42
        0xbe840380, // s_mov_b32 s4, 0
        // BB1:
        0xbe820480, // s_mov_b64 s[2:3], 0
        0xbe850380, // s_mov_b32 s5, 0
        // BB2:
        0x80060200, // s_add_u32 s6, s0, s2
        0x82070301, // s_addc_u32 s7, s1, s3
        0x81058105, // s_add_i32 s5, s5, 1
        0x80028402, // s_add_u32 s2, s2, 4
        0x82038003, // s_addc_u32 s3, s3, 0
        0xbf068605, // s_cmp_eq_u32 s5, 6
        0xdc708000, 0x00060100, // global_store_dword v0, v1, s[6:7]
        0xbf84fff7, // s_cbranch_scc0 BB2
        // BB3:
        0x81048104, // s_add_i32 s4, s4, 1
        0x80009800, // s_add_u32 s0, s0, 24
        0x82018001, // s_addc_u32 s1, s1, 0
        0xbf068504, // s_cmp_eq_u32 s4, 5
        0xbf84fff0, // s_cbranch_scc0 BB1
        // BB4:
        0xbf810000, // s_endpgm
    ];
    const KERNEL_INFO: &KernelInfo = &KernelInfo {
        name: "nested_loop",
        code: CODE,
        pgmrsrcs: [0, 0, 0],
        setup: SGPRSetup::from_bits_truncate(9),
        kern_arg_size: 8,
        kern_arg_segment_align: 0,
        group_segment_fixed_size: 0,
        private_segment_fixed_size: 0,
        arguments: vec![],
    };

    let func = Disassembler::parse_kernel(KERNEL_INFO).expect("Failed to parse the function");
    (KERNEL_INFO, func)
}

pub(crate) fn shl_operations() -> (&'static KernelInfo<'static>, Function<'static>) {
    const CODE: &[u32] = &[
        0x81088408, // s_add_i32 s8, s8, 4
        0x8f000881, // s_lshl_b32 s0, 1, s8
        0x8100c100, // s_add_i32 s0, s0, -1
        0x36000000, // v_and_b32_e32 v0, s0, v0
        0xbf810000, // s_endpgm
    ];
    const KERNEL_INFO: &KernelInfo = &KernelInfo {
        name: "shl_operations",
        code: CODE,
        pgmrsrcs: [1622081536, 144, 0],
        setup: SGPRSetup::from_bits_truncate(11),
        kern_arg_size: 8,
        kern_arg_segment_align: 0,
        group_segment_fixed_size: 0,
        private_segment_fixed_size: 0,
        arguments: vec![],
    };

    let func = Disassembler::parse_kernel(KERNEL_INFO).expect("Failed to parse the function");
    (KERNEL_INFO, func)
}

pub(crate) fn complex_nested_loop() -> (&'static KernelInfo<'static>, Function<'static>) {
    const CODE: &[u32] = &[
        // BB0:
        0xbe800380, // s_mov_b32 s0, 0
        0xbe8103b0, // s_mov_b32 s1, 48
        // BB1:
        0xbe820380, // s_mov_b32 s2, 0
        0x8703c101, // s_and_b32 s3, s1, -1
        // BB2:
        0x80028102, // s_add_u32 s2, s2, 1
        0x80038403, // s_add_u32 s3, s3, 4
        0xbf068602, // s_cmp_eq_u32 s2, 6
        0xbf84fffc, // s_cbranch_scc0 BB2
        // BB3:
        0x80008100, // s_add_u32 s0, s0, 1
        0x80019801, // s_add_u32 s1, s1, 24
        0xbf068500, // s_cmp_eq_u32 s0, 5
        0xbf84fff6, // s_cbranch_scc0 BB1
        // BB4:
        0xbf810000, // s_endpgm
    ];
    const KERNEL_INFO: &KernelInfo = &KernelInfo {
        name: "complex_nested_loop",
        code: CODE,
        pgmrsrcs: [0, 0, 0],
        setup: SGPRSetup::from_bits_truncate(9),
        kern_arg_size: 0,
        kern_arg_segment_align: 0,
        group_segment_fixed_size: 0,
        private_segment_fixed_size: 0,
        arguments: vec![],
    };
    let func = Disassembler::parse_kernel(KERNEL_INFO).expect("Failed to parse the function");
    (KERNEL_INFO, func)
}

pub(crate) fn integer_division() -> (&'static KernelInfo<'static>, Function<'static>) {
    const CODE: &[u32] = &[
        0x4a0808c0, // v_add_nc_u32_e32 v4, 64, v4
        0x800fa00f, // s_add_u32 s15, s15, 32
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
        // v0 = v4 / s15
        0xd5690000, 0x0002000f, // v_mul_lo_u32 v0, s15, v0
        0x4c000104, // v_sub_nc_u32_e32 v0, v4, v0
        // v0 = v4 % s15
        0x9b10ff0f, 0x92492493, // s_mul_hi_i32 s16, s15, 0x92492493
        0x81100f10, // s_add_i32 s16, s16, s15
        0x90009f10, // s_lshr_b32 s0, s16, 31
        0x91108510, // s_ashr_i32 s16, s16, 5
        0x81100010, // s_add_i32 s16, s16, s0
        // s16 = s15 / 56
        0x900f830f, // s_lshr_b32 s15, s15, 3
        0x9a80ff0f, 0x24924925, // s_mul_hi_u32 s0, s15, 0x24924925
        // s0 = s15 / 8 / 7
        0xbf810000, // s_endpgm
    ];
    const KERNEL_INFO: &KernelInfo = &KernelInfo {
        name: "integer_division",
        code: CODE,
        pgmrsrcs: [0, 0, 0],
        setup: SGPRSetup::from_bits_truncate(9),
        kern_arg_size: 0,
        kern_arg_segment_align: 0,
        group_segment_fixed_size: 0,
        private_segment_fixed_size: 0,
        arguments: vec![],
    };
    let func = Disassembler::parse_kernel(KERNEL_INFO).expect("Failed to parse the function");
    (KERNEL_INFO, func)
}
