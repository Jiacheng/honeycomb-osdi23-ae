use crate::ir::constraints::{Constraint, ValueDescriptor};
use crate::prover::symbolic_heap::{SymbolicLocation, SymbolicValue};
use crate::prover::SymbolicHeap;

/// Reference:
/// HSA Platform System Architecture Specification
/// Version 1.2
/// Table 2-7 Architected Queuing Language (AQL) kernel dispatch packet format
pub struct DispatchPacket {}

impl DispatchPacket {
    const WORKGROUP_SIZE_XY_OFFSET: usize = 4;
    const WORKGROUP_SIZE_Z_OFFSET: usize = 8;
    const GRID_SIZE_X_OFFSET: usize = 12;
    const GRID_SIZE_Y_OFFSET: usize = 16;
    const GRID_SIZE_Z_OFFSET: usize = 20;

    pub const PACKET_SIZE: usize = 64;

    /*
     * Register block / grid dimensions as constraints in the symbolic heap
     */
    #[allow(dead_code)]
    pub(crate) fn register_dimensions(
        heap: &mut SymbolicHeap,
        block_size: &[usize; 3],
        grid_size: &[usize; 3],
    ) {
        // The workgroup size is a ushort where the compiler usually coalesces the loads into a load_dword
        let values = [
            (
                Self::WORKGROUP_SIZE_XY_OFFSET,
                (block_size[0] as u32) | ((block_size[1] as u32) << 16),
            ),
            (Self::WORKGROUP_SIZE_Z_OFFSET, block_size[1] as u32),
            (
                Self::GRID_SIZE_X_OFFSET,
                (grid_size[0] * block_size[0]) as u32,
            ),
            (
                Self::GRID_SIZE_Y_OFFSET,
                (grid_size[1] * block_size[1]) as u32,
            ),
            (
                Self::GRID_SIZE_Z_OFFSET,
                (grid_size[2] * block_size[2]) as u32,
            ),
        ];
        for (offset, value) in values.into_iter() {
            let v = SymbolicValue {
                len: 4,
                value: ValueDescriptor::Value(Constraint::singleton(value as isize)),
            };
            heap.register_symbolic_location_constraint(SymbolicLocation::DispatchPacket(offset), v);
        }
    }
}
