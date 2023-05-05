use crate::fileformat::{Disassembler, KernelInfo, SGPRSetup};
use crate::ir::constraints::{Constraint, ValueDescriptor};
use crate::ir::Function;
use crate::prover::symbolic_heap::{SymbolicLocation, SymbolicValue};
use crate::prover::SymbolicHeap;

#[derive(Debug)]
pub(crate) struct MemoryAnalysisTestCase {
    pub(crate) kernel: &'static KernelInfo<'static>,
    pub(crate) func: Function<'static>,
    pub(crate) constraints: &'static [(SymbolicLocation, SymbolicValue)],
}

impl MemoryAnalysisTestCase {
    pub(crate) fn register_constraints(&self, heap: &mut SymbolicHeap) {
        for (loc, val) in self.constraints {
            heap.register_symbolic_location_constraint(loc.clone(), val.clone())
        }
    }
}

pub(crate) fn load_arguments() -> MemoryAnalysisTestCase {
    const CODE: &[u32] = &[
        0xf4000002, 0xfa000004, // s_load_dword s0, s[4:5], 0x4
        0xf4080002, 0xfa000008, // s_load_dwordx4 s[0:3], s[4:5], 0x8
        0xf4000003, 0xfa000000, // s_load_dword s0, s[6:7], 0x0
        0xf4000003, 0xfa000004, // s_load_dword s0, s[6:7], 0x4
        0xbf810000, // s_endpgm
    ];
    const KERNEL_INFO: &KernelInfo = &KernelInfo {
        name: "load_argument",
        code: CODE,
        pgmrsrcs: [1622081536, 144, 0],
        setup: SGPRSetup::from_bits_truncate(11),
        kern_arg_size: 8,
        kern_arg_segment_align: 0,
        group_segment_fixed_size: 0,
        private_segment_fixed_size: 0,
        arguments: vec![],
    };
    const CONSTRAINTS: &[(SymbolicLocation, SymbolicValue)] = &[
        (
            SymbolicLocation::KernelArgBuffer(0),
            SymbolicValue {
                len: 4,
                value: ValueDescriptor::Value(Constraint { min: 100, max: 200 }),
            },
        ),
        (
            SymbolicLocation::KernelArgBuffer(4),
            SymbolicValue {
                len: 4,
                value: ValueDescriptor::Value(Constraint { min: 300, max: 400 }),
            },
        ),
    ];

    let func = Disassembler::parse_kernel(KERNEL_INFO).expect("Failed to parse the function");
    MemoryAnalysisTestCase {
        kernel: KERNEL_INFO,
        func,
        constraints: CONSTRAINTS,
    }
}

pub(crate) fn load_argument_value() -> MemoryAnalysisTestCase {
    const CODE: &[u32] = &[
        0xf4080003, 0xfa000000, // s_load_dwordx4 s[0:3], s[6:7], 0x0
        0xf4000003, 0xfa000008, // s_load_dword s0, s[6:7], 0x8
        0xbf810000, // s_endpgm
    ];
    const KERNEL_INFO: &KernelInfo = &KernelInfo {
        name: "load_argument_value",
        code: CODE,
        pgmrsrcs: [1622081536, 144, 0],
        setup: SGPRSetup::from_bits_truncate(11),
        kern_arg_size: 16,
        kern_arg_segment_align: 0,
        group_segment_fixed_size: 0,
        private_segment_fixed_size: 0,
        arguments: vec![],
    };
    const CONSTRAINTS: &[(SymbolicLocation, SymbolicValue)] = &[
        (
            SymbolicLocation::KernelArgBuffer(0),
            SymbolicValue {
                len: 4,
                value: ValueDescriptor::Value(Constraint { min: 100, max: 200 }),
            },
        ),
        (
            SymbolicLocation::KernelArgBuffer(8),
            SymbolicValue {
                len: 8,
                value: ValueDescriptor::Value(Constraint { min: 300, max: 400 }),
            },
        ),
    ];

    let func = Disassembler::parse_kernel(KERNEL_INFO).expect("Failed to parse the function");
    MemoryAnalysisTestCase {
        kernel: KERNEL_INFO,
        func,
        constraints: CONSTRAINTS,
    }
}
