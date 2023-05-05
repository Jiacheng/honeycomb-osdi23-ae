/**
 * Describe the pre-conditions of various memory locations before launching the kernels
 **/
#[derive(Copy, Clone, Debug)]
pub struct MemoryConstraint {
    pub loc: Location,
    pub constraint: Constraint,
}

#[derive(Copy, Clone, Debug)]
pub enum Location {
    // A kernel argument that is a pointer
    KernelArgumentPointer(usize),
}

// Only support min-max for now
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Constraint {
    pub min: isize,
    pub max: isize,
}

impl Constraint {
    pub fn singleton(value: isize) -> Self {
        Self {
            min: value,
            max: value,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[allow(dead_code)]
pub(crate) enum ValueDescriptor {
    // Pointer to the RO heap
    SystemPointer(Constraint),
    // Pointer to the RW heap
    HeapPointer(Constraint),
    // A concrete value
    Value(Constraint),
}

impl MemoryConstraint {
    pub fn new(loc: Location, constraint: Constraint) -> Self {
        Self { loc, constraint }
    }
}
