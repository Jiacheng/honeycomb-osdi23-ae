use std::fmt::{Display, Formatter};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("ELF parsing error: {0}")]
    ELFError(#[from] goblin::error::Error),
    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),
    #[error("Decode error")]
    DecodeError(#[from] DecodeError),
    #[error("Validation error: {0}")]
    ValidationError(#[from] ValidationError),
}

#[derive(Debug)]
pub enum DecodeError {
    InvalidOperand,
    InvalidInstruction,
    InvalidOpcode,
    InvalidDescriptor,
    InvalidModifier,
}

#[derive(Debug)]
pub enum ValidationError {
    UnresolvedAddress,
    OutOfBounds(usize, usize),
    InvalidAccess,
    UnexpectedKernelArgument(usize),
}

impl std::error::Error for DecodeError {}

impl Display for DecodeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::InvalidOperand => f.pad("Invalid operand"),
            DecodeError::InvalidInstruction => f.pad("Invalid instruction"),
            DecodeError::InvalidOpcode => f.pad("Invalid opcode"),
            DecodeError::InvalidDescriptor => f.pad("Invalid descriptor"),
            DecodeError::InvalidModifier => f.pad("Invalid modifier"),
        }
    }
}

impl std::error::Error for ValidationError {}

impl Display for ValidationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::UnresolvedAddress => f.pad("Unresolved address"),
            ValidationError::OutOfBounds(actual, bound) => {
                write!(f, "out of bounds {} >= {}", actual, bound)
            }
            ValidationError::InvalidAccess => f.pad("Invalid access"),
            ValidationError::UnexpectedKernelArgument(off) => {
                write!(f, "unexpected kernel argument at offset {}", off)
            }
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
