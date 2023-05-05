use crate::analysis::SCEVExprRef;
use crate::error::{Error, Result, ValidationError};
use crate::fileformat::ArgInfo;
use crate::ir::constraints::ValueDescriptor::HeapPointer;
use crate::ir::constraints::{Constraint, Location, MemoryConstraint, ValueDescriptor};
use crate::ir::ImplicitLoopInfo;
use std::collections::{BTreeMap, HashMap};

/**
 * A symbolic heap that represents the read-only portion of the memory address space.
 * The typical usage is to track the values and constraints for the kernel arguments.
 **/
#[derive(Default, Debug)]
pub struct SymbolicHeap {
    heap: BTreeMap<AbstractLocation, SymbolicValue>,
    // kernel arg offset -> abs loc
    // dispatch packet offset -> abs loc
    // the value of the kernel arguments -> abs loc
    // To ensure soundness aliasing is restricted within the corresponding regions, (i.e., a RW pointer will never be aliased to a RO pointer)
    location_map: HashMap<SymbolicLocation, AbstractLocation>,
    abs_loc_id: usize,
}

type AbstractLocation = usize;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) enum SymbolicLocation {
    // Accessing the buffer of kernel arguments, *(kernel_arg_base + offset)
    KernelArgBuffer(usize),
    // Accessing the dispatch packet, *(dispatch_packet_ptr + offset)
    DispatchPacket(usize),
    // Accessing the buffer specified in kernel arguments, i.e., **(kernel_arg_base + offset)
    Heap(SCEVExprRef),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SymbolicValue {
    pub len: usize,
    pub value: ValueDescriptor,
}

impl SymbolicValue {
    fn from_argument_constraint(arg_info: &ArgInfo, m: &MemoryConstraint) -> Self {
        Self {
            len: arg_info.length,
            value: HeapPointer(m.constraint),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum AccessDescriptor {
    Bit32,
    Bit64,
    Low32Of64,
}

impl AccessDescriptor {
    /// Build the descriptor for a symbolic value if it is wholy or partially accessed
    fn check(
        access_addr: usize,
        access_len: usize,
        symbolic_addr: usize,
        symbolic_value: &SymbolicValue,
    ) -> Option<(Self, usize, SymbolicValue)> {
        let symbolic_len = symbolic_value.len;
        let intersect_l = usize::max(access_addr, symbolic_addr);
        let intersect_r = usize::min(access_addr + access_len, symbolic_addr + symbolic_len);
        if intersect_l >= intersect_r {
            // no intersection. the symbolic value is not accessed.
            return None;
        }
        let intersect_len = intersect_r - intersect_l;
        let offset_in_symbolic = intersect_l - symbolic_addr;
        let offset_in_accessed = intersect_l - access_addr;
        match (offset_in_symbolic, intersect_len, symbolic_value.len) {
            (0, 4, 4) if offset_in_accessed % 4 == 0 => {
                // wholy accessed 32bit value
                Some((Self::Bit32, offset_in_accessed, symbolic_value.clone()))
            }
            (0, 8, 8) if offset_in_accessed % 4 == 0 => {
                // wholy accessed 64bit value
                Some((Self::Bit64, offset_in_accessed, symbolic_value.clone()))
            }
            (0, 4, 8) if offset_in_accessed % 4 == 0 => {
                // 64bit value whose low-32-bits are accessed
                Some((Self::Low32Of64, offset_in_accessed, symbolic_value.clone()))
            }
            _ => None,
        }
    }
}

impl SymbolicHeap {
    pub fn new() -> Self {
        SymbolicHeap::default()
    }

    pub fn load_constraints(
        &mut self,
        m: &MemoryConstraint,
        args: &HashMap<usize, &ArgInfo>,
    ) -> Result<()> {
        match m.loc {
            Location::KernelArgumentPointer(off) => {
                let arg_info = args.get(&off).ok_or(Error::ValidationError(
                    ValidationError::UnexpectedKernelArgument(off),
                ))?;
                let v = SymbolicValue::from_argument_constraint(arg_info, m);
                self.register_argument_constraint(off, v);
            }
        }
        Ok(())
    }

    pub(crate) fn new_loc(&mut self) -> usize {
        let ret = self.abs_loc_id;
        self.abs_loc_id += 1;
        ret
    }

    pub(crate) fn register_argument_constraint(&mut self, offset: usize, value: SymbolicValue) {
        self.register_symbolic_location_constraint(SymbolicLocation::KernelArgBuffer(offset), value)
    }

    pub(crate) fn register_dispatch_packet_constraint(
        &mut self,
        offset: usize,
        value: SymbolicValue,
    ) {
        self.register_symbolic_location_constraint(SymbolicLocation::DispatchPacket(offset), value)
    }

    pub(crate) fn register_ro_load(&mut self, load_result: SCEVExprRef, value: SymbolicValue) {
        self.register_symbolic_location_constraint(SymbolicLocation::Heap(load_result), value)
    }

    pub(crate) fn register_implicit_loop_bounds(&mut self, shape: &ImplicitLoopInfo) {
        // According to http://hsafoundation.com/wp-content/uploads/2021/02/HSA-SysArch-1.2.pdf Page 29 Table 2-7
        const WORKGROUP_SIZE_XY_OFFSET: usize = 4;
        const WORKGROUP_SIZE_Z_OFFSET: usize = 8;
        const GRID_SIZE_X_OFFSET: usize = 12;
        const GRID_SIZE_Y_OFFSET: usize = 16;
        const GRID_SIZE_Z_OFFSET: usize = 20;
        let &dim = &shape.dimensions;
        // All workgroup sizes are 16-bit word. The compiler always promotes the loads as 32-bit loads.
        // Notably the compiler promotes the load of workgroup_size_z to a 32-bit load and an and instruction
        // to get the value of workgroup_size_z.
        let values = [
            (
                WORKGROUP_SIZE_XY_OFFSET,
                (dim[0].workgroup_size as u32) | ((dim[1].workgroup_size as u32) << 16),
            ),
            (WORKGROUP_SIZE_Z_OFFSET, dim[2].workgroup_size as u32),
            (
                GRID_SIZE_X_OFFSET,
                (dim[0].grid_size * dim[0].workgroup_size) as u32,
            ),
            (
                GRID_SIZE_Y_OFFSET,
                (dim[1].grid_size * dim[1].workgroup_size) as u32,
            ),
            (
                GRID_SIZE_Z_OFFSET,
                (dim[2].grid_size * dim[2].workgroup_size) as u32,
            ),
        ];
        for (offset, value) in values.into_iter() {
            let v = SymbolicValue {
                len: 4,
                value: ValueDescriptor::Value(Constraint::singleton(value as isize)),
            };
            self.register_dispatch_packet_constraint(offset, v);
        }
    }

    pub(crate) fn register_symbolic_location_constraint(
        &mut self,
        sym_loc: SymbolicLocation,
        value: SymbolicValue,
    ) {
        let loc = self.new_loc();
        self.location_map.insert(sym_loc, loc);
        self.heap.insert(loc, value);
    }

    /**
     * Query the constraints on an address on the heap. There might be multiple constraints available for the specified length due to load coalescing.
     *
     **/
    pub(crate) fn get(
        &self,
        addr: &SymbolicLocation,
        len: usize,
    ) -> Option<Vec<(AccessDescriptor, usize, SymbolicValue)>> {
        match addr {
            SymbolicLocation::KernelArgBuffer(u) | SymbolicLocation::DispatchPacket(u) => {
                let r = self
                    .location_map
                    .iter()
                    .filter_map(|(offset, loc)| match offset {
                        SymbolicLocation::KernelArgBuffer(start)
                            if matches!(addr, SymbolicLocation::KernelArgBuffer(_)) =>
                        {
                            AccessDescriptor::check(*u, len, *start, self.heap.get(loc)?)
                        }
                        SymbolicLocation::DispatchPacket(start)
                            if matches!(addr, SymbolicLocation::DispatchPacket(_)) =>
                        {
                            AccessDescriptor::check(*u, len, *start, self.heap.get(loc)?)
                        }
                        _ => None,
                    })
                    .collect::<Vec<(AccessDescriptor, usize, SymbolicValue)>>();
                Some(r)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fileformat::KernelArgValueKind;
    use crate::ir::constraints::Constraint;

    #[test]
    fn test_implicit_loop_bounds() {
        let mut heap = SymbolicHeap::new();
        heap.register_implicit_loop_bounds(&ImplicitLoopInfo::new([256, 16, 4], [64, 8, 1]));
        [
            (4, 256 + 16 * 0x10000),
            (8, 4),
            (12, 256 * 64),
            (16, 16 * 8),
            (20, 4 * 1),
        ]
        .iter()
        .cloned()
        .for_each(|(offset, v)| {
            let location = SymbolicLocation::DispatchPacket(offset);
            let values = heap.get(&location, 4).expect("Can not get value");
            assert_eq!(
                vec![(
                    AccessDescriptor::Bit32,
                    0,
                    SymbolicValue {
                        value: ValueDescriptor::Value(Constraint::singleton(v)),
                        len: 4,
                    }
                )],
                values
            );
        });
    }

    #[test]
    fn test_load_constraints() {
        let mut heap = SymbolicHeap::new();
        /*
         *  .offset:         0
         *  .size:           8
         *  .type_name:      'float*'
         *  .value_kind:     global_buffer
         *
         *- .offset:         8
         *  .size:           4
         *  .type_name:      float
         *  .value_kind:     by_value
         *
         *  .offset:         12
         *  .size:           4
         *  .type_name:      int
         *  .value_kind:     global_buffer
         *- .address_space:  global
         *
         *- .offset:         16
         *  .size:           8
         *  .value_kind:     hidden_global_offset_x
         *
         *- .offset:         24
         *  .size:           8
         *  .value_kind:     hidden_global_offset_y
         *
         *- .offset:         32
         *  .size:           8
         *  .value_kind:     hidden_global_offset_z
         *- .address_space:  global
         *
         *  .offset:         40
         *  .size:           8
         *  .value_kind:     hidden_multigrid_sync_arg
         */
        let arguments_map = [
            ArgInfo {
                name: None,
                offset: 0,
                length: 8,
                value_kind: KernelArgValueKind::ByValue,
            },
            ArgInfo {
                name: None,
                offset: 8,
                length: 4,
                value_kind: KernelArgValueKind::ByValue,
            },
            ArgInfo {
                name: None,
                offset: 12,
                length: 4,
                value_kind: KernelArgValueKind::ByValue,
            },
            ArgInfo {
                name: None,
                offset: 16,
                length: 8,
                value_kind: KernelArgValueKind::HiddenGlobalOffsetX,
            },
            ArgInfo {
                name: None,
                offset: 24,
                length: 8,
                value_kind: KernelArgValueKind::HiddenGlobalOffsetY,
            },
            ArgInfo {
                name: None,
                offset: 32,
                length: 8,
                value_kind: KernelArgValueKind::HiddenGlobalOffsetZ,
            },
            ArgInfo {
                name: None,
                offset: 40,
                length: 8,
                value_kind: KernelArgValueKind::HiddenMultiGridSyncArg,
            },
        ]
        .iter()
        .map(|arg_info| (arg_info.offset, arg_info))
        .collect::<HashMap<usize, &ArgInfo>>();
        let expected = [(0, 8), (8, 4), (12, 4), (16, 8), (24, 8), (32, 8), (40, 8)];
        let result = expected.iter().try_for_each(|(off, _len)| {
            heap.load_constraints(
                &MemoryConstraint::new(
                    Location::KernelArgumentPointer(*off),
                    Constraint::singleton(0),
                ),
                &arguments_map,
            )
        });
        assert!(result.is_ok());
        expected.iter().for_each(|(off, len)| {
            let values = heap
                .get(&SymbolicLocation::KernelArgBuffer(*off), *len)
                .expect("Can not get value");
            if *len == 4 {
                assert_eq!(
                    values,
                    vec![(
                        AccessDescriptor::Bit32,
                        0,
                        SymbolicValue {
                            value: ValueDescriptor::HeapPointer(Constraint::singleton(0)),
                            len: *len,
                        }
                    )]
                );
            } else if *len == 8 {
                assert_eq!(
                    values,
                    vec![(
                        AccessDescriptor::Bit64,
                        0,
                        SymbolicValue {
                            value: ValueDescriptor::HeapPointer(Constraint::singleton(0)),
                            len: *len,
                        }
                    )]
                );
            }
        });
    }
}
