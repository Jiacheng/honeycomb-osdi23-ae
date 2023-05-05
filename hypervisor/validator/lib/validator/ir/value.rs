use smallvec::SmallVec;
use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};

#[derive(Copy, Clone, Debug)]
pub enum APConstant {
    ConstantInt(isize),
    ConstantFloat(f64),
}

/**
 * A representation of the definition of a value. It is okay to store multiple copies of the values since it is a representation.
 **/
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Value {
    Undefined,
    Argument(usize),
    // The indices of the instruction and the operand
    Instruction(usize, usize),
    Phi(usize),
    Constant(APConstant),
}

impl PartialEq<Self> for APConstant {
    fn eq(&self, other: &Self) -> bool {
        use APConstant::*;
        match (self, other) {
            (ConstantInt(x), ConstantInt(y)) => *x == *y,
            (ConstantFloat(x), ConstantFloat(y)) => x.to_bits() == y.to_bits(),
            _ => false,
        }
    }
}

impl APConstant {
    pub(crate) fn opposite(&self) -> Self {
        match self {
            Self::ConstantFloat(x) => Self::ConstantFloat(-x),
            Self::ConstantInt(x) => Self::ConstantInt(-x),
        }
    }
}

impl Eq for APConstant {}

impl Hash for APConstant {
    fn hash<H: Hasher>(&self, state: &mut H) {
        use APConstant::*;
        match self {
            ConstantInt(x) => {
                x.hash(state);
            }
            ConstantFloat(x) => {
                1.hash(state);
                x.to_bits().hash(state);
            }
        }
    }
}

#[derive(Clone, Eq, PartialEq, Hash, Default)]
pub struct PHI {
    // The basic block that the phi node belongs to
    pub bb_idx: usize,
    // Incoming values and the indices of the respective basic block
    pub values: SmallVec<[(usize, Box<Value>); 2]>,
}

impl PHI {
    pub fn new(bb_idx: usize, values: SmallVec<[(usize, Box<Value>); 2]>) -> PHI {
        Self { bb_idx, values }
    }

    pub fn get_incoming_value_for_block(&self, block_id: usize) -> Option<&Value> {
        self.values
            .iter()
            .find(|(b, _)| *b == block_id)
            .map(|x| x.1.as_ref())
    }
}

impl Debug for PHI {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "PHI({}) [", self.bb_idx)?;
        for (incoming, v) in self.values.iter() {
            write!(f, "[{}, {:#?}]", incoming, v.as_ref())?;
        }
        write!(f, "]")
    }
}
