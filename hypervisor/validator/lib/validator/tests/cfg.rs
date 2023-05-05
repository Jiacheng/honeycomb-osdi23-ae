use crate::fileformat::{Disassembler, KernelInfo, SGPRSetup};
use crate::ir::Function;

pub(crate) const fn simple_kernel_info<'a>(name: &'a str, code: &'a [u32]) -> KernelInfo<'a> {
    KernelInfo {
        name,
        code,
        pgmrsrcs: [0, 0, 0],
        setup: SGPRSetup::from_bits_truncate(9),
        kern_arg_size: 0,
        kern_arg_segment_align: 0,
        group_segment_fixed_size: 0,
        private_segment_fixed_size: 0,
        arguments: vec![],
    }
}

pub(crate) fn nested_loop() -> (&'static KernelInfo<'static>, Function<'static>) {
    // s_load_dwordx2 s[0:1], s[4:5], 0x0                         // 000000001000: F4040002 FA000000
    // v_mov_b32_e32 v0, 0                                        // 000000001008: 7E000280
    // v_mov_b32_e32 v1, 42                                       // 00000000100C: 7E0202AA
    // s_mov_b32 s4, 0                                            // 000000001010: BE840380
    // s_mov_b64 s[2:3], 0                                        // 000000001014: BE820480
    // s_waitcnt lgkmcnt(0)                                       // 000000001018: BF8CC07F
    // s_add_u32 s6, s0, s2                                       // 00000000101C: 80060200
    // s_addc_u32 s7, s1, s3                                      // 000000001020: 82070301
    // s_add_u32 s2, s2, 4                                        // 000000001024: 80028402
    // s_addc_u32 s3, s3, 0                                       // 000000001028: 82038003
    // s_cmp_eq_u32 s2, 24                                        // 00000000102C: BF069802
    // global_store_dword v0, v1, s[6:7]                          // 000000001030: DC708000 00060100
    // s_cbranch_scc0 65527                                       // 000000001038: BF84FFF7 <f+0x18>
    // s_add_i32 s4, s4, 1                                        // 00000000103C: 81048104
    // s_add_u32 s0, s0, 24                                       // 000000001040: 80009800
    // s_addc_u32 s1, s1, 0                                       // 000000001044: 82018001
    // s_cmp_eq_u32 s4, 5                                         // 000000001048: BF068504
    // s_cbranch_scc0 65521                                       // 00000000104C: BF84FFF1 <f+0x14>
    // s_endpgm                                                   // 000000001050: BF810000
    const CODE: &[u32] = &[
        0xf4040002, 0xfa000000, 0x7e000280, 0x7e0202aa, 0xbe840380, 0xbe820480, 0xbf8cc07f,
        0x80060200, 0x82070301, 0x80028402, 0x82038003, 0xbf069802, 0xdc708000, 0x00060100,
        0xbf84fff7, 0x81048104, 0x80009800, 0x82018001, 0xbf068504, 0xbf84fff1, 0xbf810000,
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

pub(crate) fn double_step_loop() -> (&'static KernelInfo<'static>, Function<'static>) {
    // s_mov_b32 s0, 0                                            // 000000001000: BE800380
    // s_add_u32 s0, s0, 1                                        // 000000001004: 80008100
    // s_add_u32 s0, s0, 1                                        // 000000001008: 80008100
    // s_cmp_lg_u32 s0, 8                                         // 00000000100C: BF078800
    // s_cbranch_scc1 65532                                       // 000000001010: BF85FFFC <f+0x4>
    // s_endpgm                                                   // 000000001014: BF810000
    const CODE: &[u32] = &[
        0xBE800380, 0x80008100, 0x80008100, 0xBF078800, 0xBF85FFFC, 0xBF810000,
    ];
    const KERNEL_INFO: &KernelInfo = &simple_kernel_info("double_step_loop", &CODE);
    let func = Disassembler::parse_kernel(KERNEL_INFO).expect("Failed to parse the function");
    (KERNEL_INFO, func)
}

pub(crate) fn clear_cp() -> (&'static KernelInfo<'static>, Function<'static>) {
    // s_load_dwordx2 s[0:1], s[4:5], null                        // 000000001000: F4040002 FA000000
    // v_mov_b32_e32 v1, 0                                        // 000000001008: 7E020280
    // v_lshl_add_u32 v0, s6, 8, v0                               // 00000000100C: D7460000 04011006
    // v_mov_b32_e32 v2, 42                                       // 000000001014: 7E0402AA
    // v_lshlrev_b64 v[0:1], 2, v[0:1]                            // 000000001018: D6FF0000 00020082
    // s_waitcnt lgkmcnt(0)                                       // 000000001020: BF8CC07F
    // v_add_co_u32 v0, vcc_lo, s0, v0                            // 000000001024: D70F6A00 00020000
    // v_add_co_ci_u32_e32 v1, vcc_lo, s1, v1, vcc_lo             // 00000000102C: 50020201
    // global_store_dword v[0:1], v2, off                         // 000000001030: DC708000 007D0200
    // s_endpgm                                                   // 000000001038: BF810000
    const CODE: &[u32] = &[
        0xF4040002, 0xFA000000, 0x7E020280, 0xD7460000, 0x04011006, 0x7E0402AA, 0xD6FF0000,
        0x00020082, 0xBF8CC07F, 0xD70F6A00, 0x00020000, 0x50020201, 0xDC708000, 0x007D0200,
        0xBF810000,
    ];
    const KERNEL_INFO: &KernelInfo = &KernelInfo {
        name: "clear_cp",
        code: CODE,
        pgmrsrcs: [0x60af0040, 0x8c, 0],
        setup: SGPRSetup::from_bits_truncate(9),
        kern_arg_size: 8,
        kern_arg_segment_align: 8,
        group_segment_fixed_size: 0,
        private_segment_fixed_size: 0,
        arguments: vec![],
    };
    let func = Disassembler::parse_kernel(KERNEL_INFO).expect("Failed to parse the function");
    (KERNEL_INFO, func)
}

pub(crate) fn store_at_grid_size() -> (&'static KernelInfo<'static>, Function<'static>) {
    // s_load_dword s0, s[4:5], 0xc                               // 000000001000: F4000002 FA00000C
    // s_load_dwordx2 s[2:3], s[6:7], null                        // 000000001008: F4040083 FA000000
    // s_mov_b32 s1, 0                                            // 000000001010: BE810380
    // v_mov_b32_e32 v0, 0                                        // 000000001014: 7E000280
    // v_mov_b32_e32 v1, 42                                       // 000000001018: 7E0202AA
    // s_waitcnt lgkmcnt(0)                                       // 00000000101C: BF8CC07F
    // s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001020: 8F808200
    // s_add_u32 s0, s2, s0                                       // 000000001024: 80000002
    // s_addc_u32 s1, s3, s1                                      // 000000001028: 82010103
    // global_store_dword v0, v1, s[0:1]                          // 00000000102C: DC708000
    const CODE: &[u32] = &[
        0xF4000002, 0xFA00000C, 0xF4040083, 0xFA000000, 0xBE810380, 0x7E000280, 0x7E0202AA,
        0xBF8CC07F, 0x8F808200, 0x80000002, 0x82010103, 0xDC708000,
    ];
    const KERNEL_INFO: &KernelInfo = &KernelInfo {
        name: "",
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

pub(crate) fn vgpr_induction_loop() -> (&'static KernelInfo<'static>, Function<'static>) {
    // v_mov_b32 v0, 0                                            // 000000001010: 7E000280
    // s_mov_b32 s0, 0                                            // 000000001014: BE800380
    // v_add_nc_u32_e32 v0, 1, v0                                 // 000000001018: 4A000081
    // s_nop 0                                                    // 00000000101C: BF800000
    // v_cmp_lt_u32_e32 vcc_lo, 0x7f, v0                          // 000000001020: 7D8200FF 0000007F
    // s_or_b32 s0, vcc_lo, s0                                    // 000000001028: 8800006A
    // s_andn2_b32 exec_lo, exec_lo, s0                           // 00000000102C: 8A7E007E
    // s_cbranch_execnz 65529                                     // 000000001030: BF89FFF9
    // s_endpgm                                                   // 000000001034: BF810000
    const CODE: &[u32] = &[
        0x7E000280, 0xBE800380, 0x4A000081, 0xBF800000, 0x7D8200FF, 0x0000007F, 0x8800006A,
        0x8A7E007E, 0xBF89FFF9, 0xBF810000,
    ];
    const KERNEL_INFO: &KernelInfo = &simple_kernel_info("vgpr_induction_loop", &CODE);
    let func = Disassembler::parse_kernel(KERNEL_INFO).expect("Failed to parse the function");
    (KERNEL_INFO, func)
}
