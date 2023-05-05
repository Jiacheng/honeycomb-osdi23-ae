pub mod constraints;
mod dom;
pub mod instruction;
pub mod machine;
mod runtime_info;
mod value;

use crate::isa::rdna2::Instruction;
use smallvec::SmallVec;
use std::ops::Range;

use crate::fileformat::KernelInfo;
pub use dom::{DomTree, CFG};
pub(crate) use runtime_info::ImplicitLoopInfo;
pub use runtime_info::RuntimeInfo;
pub use value::{APConstant, Value, PHI};

/**
 * A basic block.
 * The parent (i.e., Kernel) owns a number of arenas and the BasicBlock
 * only stores the respective indices.
 * The parent is expected to be read-only and the callee is expected to pass in the correct parent every time :-(
 **/
#[derive(Clone, Debug)]
pub struct BasicBlock {
    /** Relative offset to the start of the kernel in dwords */
    pub offset: usize,
    pub(crate) predecessors: SmallVec<[usize; 4]>,
    pub(crate) successors: SmallVec<[usize; 4]>,
    pub instructions: Range<usize>,
}

#[derive(Clone, Debug)]
pub struct Function<'a> {
    pub name: &'a str,
    pub basic_blocks: Vec<BasicBlock>,
    pub instructions: Vec<Instruction>,
}

#[derive(Clone)]
pub struct Module<'a> {
    pub kernels: Vec<(KernelInfo<'a>, Function<'a>)>,
}

/**
 * Primitive types of a value.
 **/
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Type {
    Int32,
    Int64,
    Float32,
    Float64,
    Unknown,
}

impl Type {
    pub fn is_integer_ty(self) -> bool {
        matches!(self, Type::Int32 | Type::Int64)
    }
}

impl BasicBlock {
    pub const ENTRY_INDEX: usize = 0;
    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() > 0
    }

    pub fn get_predecessors<'a>(&self, parent: &'a Function<'a>, idx: usize) -> &'a BasicBlock {
        &parent.basic_blocks[self.predecessors[idx]]
    }

    pub fn get_successors<'a>(&self, parent: &'a Function<'a>, idx: usize) -> &'a BasicBlock {
        &parent.basic_blocks[self.successors[idx]]
    }

    pub fn instructions<'a>(&self, parent: &'a Function<'a>) -> &'a [Instruction] {
        &parent.instructions[self.instructions.clone()]
    }
}
