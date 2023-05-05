use crate::isa::rdna2::isa::Operand;
use crate::support::add_u8_i8;
use core::cmp::PartialEq;
use core::fmt::{Debug, Formatter};
use core::hash::{Hash, Hasher};

/**
 * Represent a single register in the instruction, which is the basic unit of the analysis.
 **/
#[derive(Copy, Clone, Ord, PartialOrd, Eq)]
pub enum Register {
    Scalar(u8),
    Vector(u8),
    //
    // TODO: Map registers like scc / vcc / exec to concrete SGPR before hand to reason about aliasing.
    // Eventually we should only see M0 here
    SpecialSGPR(u8),
}

impl Debug for Register {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Register::Scalar(idx) => write!(f, "s{}", idx),
            Register::Vector(idx) => write!(f, "v{}", idx),
            Register::SpecialSGPR(idx) => {
                match Operand::SPECIAL_REG_NAMES
                    .iter()
                    .find(|(id, _)| *idx == *id)
                {
                    None => write!(f, "sp{}", idx),
                    Some((_, v)) => f.pad(v),
                }
            }
        }
    }
}

impl Register {
    pub(crate) fn from_operand(op: &Operand) -> Option<(Register, u8)> {
        use crate::isa::rdna2::isa::Operand::*;
        match op {
            ScalarRegister(start, len) => Some((Register::Scalar(*start), *len)),
            SpecialScalarRegister(idx) => Some((Register::SpecialSGPR(*idx), 1)),
            VectorRegister(start, len) => Some((Register::Vector(*start), *len)),
            _ => None,
        }
    }

    fn get_idx(&self) -> u8 {
        match self {
            Register::Scalar(x) => *x,
            Register::Vector(x) => *x,
            Register::SpecialSGPR(x) => *x,
        }
    }

    fn get_repr(&self) -> u32 {
        let ty = match self {
            Register::Scalar(_) => 0,
            Register::Vector(_) => 1,
            Register::SpecialSGPR(_) => 2,
        } as u32;
        (ty << 8) | (self.get_idx() as u32)
    }

    pub(crate) fn offset(&self, off: i8) -> Register {
        use Register::*;
        match self {
            Scalar(idx) => Scalar(add_u8_i8(*idx, off)),
            Vector(idx) => Vector(add_u8_i8(*idx, off)),
            SpecialSGPR(_) => {
                if off == 0 {
                    *self
                } else {
                    unreachable!()
                }
            }
        }
    }
}

impl Hash for Register {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.get_repr().hash(state)
    }
}

impl PartialEq<Self> for Register {
    fn eq(&self, other: &Self) -> bool {
        self.get_repr() == other.get_repr()
    }
}
