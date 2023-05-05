mod global_memory;
pub(crate) mod scalar_range;
pub(crate) mod symbolic_heap;

pub use global_memory::prove;
pub use symbolic_heap::SymbolicHeap;
