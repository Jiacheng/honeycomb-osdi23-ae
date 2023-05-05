use nom_locate::LocatedSpan;
use std::fmt::{Display, Formatter};
use std::{fmt, io};

pub type Span<'a> = LocatedSpan<&'a str, &'a str>;
pub type Token<'a> = Span<'a>;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    IoError(#[from] io::Error),
    #[error("Format error: {0}")]
    FmtError(#[from] fmt::Error),
    #[error("Syntax error: {0}")]
    SyntaxError(String),
    #[error("Semantic error: {0}")]
    SemanticError(SemanticError),
}

#[derive(Default, Debug, Clone)]
pub struct Location {
    pub line: u32,
    pub offset: usize,
    pub file: String,
}

#[derive(Debug)]
pub enum SemanticErrorKind {
    UnresolvedSymbol,
    InvalidType,
    DuplicatedSymbol,
    MismatchArgument,
    Unimplemented,
}

#[derive(Debug)]
pub struct SemanticError {
    pub loc: Option<Location>,
    pub kind: SemanticErrorKind,
    pub desc: String,
}

impl<'a> From<Token<'a>> for Location {
    fn from(e: Token<'a>) -> Self {
        Self {
            line: e.location_line(),
            offset: e.get_utf8_column(),
            file: e.extra.to_string(),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

impl SemanticError {
    pub fn unimplemented(desc: String) -> Error {
        Error::SemanticError(SemanticError {
            loc: None,
            kind: SemanticErrorKind::Unimplemented,
            desc,
        })
    }

    pub fn unresolved_symbol(desc: String) -> Error {
        Error::SemanticError(SemanticError {
            loc: None,
            kind: SemanticErrorKind::UnresolvedSymbol,
            desc,
        })
    }

    pub fn unexpected_type(desc: String) -> Error {
        Error::SemanticError(SemanticError {
            loc: None,
            kind: SemanticErrorKind::InvalidType,
            desc,
        })
    }

    pub fn mismatched_argument(desc: String) -> Error {
        Error::SemanticError(SemanticError {
            loc: None,
            kind: SemanticErrorKind::MismatchArgument,
            desc,
        })
    }

    pub fn redefinition(desc: String) -> Error {
        Error::SemanticError(SemanticError {
            loc: None,
            kind: SemanticErrorKind::DuplicatedSymbol,
            desc,
        })
    }
}

impl Display for SemanticError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if let Some(l) = &self.loc {
            f.write_fmt(format_args!("{}:{}:{}", l.file, l.line, l.offset))?;
        }
        match self.kind {
            SemanticErrorKind::UnresolvedSymbol => {
                f.write_fmt(format_args!("unresolved symbol {}", self.desc))?
            }
            SemanticErrorKind::InvalidType => {
                f.write_fmt(format_args!("expecting type {}", self.desc))?
            }
            SemanticErrorKind::DuplicatedSymbol => {
                f.write_fmt(format_args!("redefinition {}", self.desc))?
            }
            SemanticErrorKind::MismatchArgument => {
                f.write_fmt(format_args!("invalid argument {}", self.desc))?
            }
            SemanticErrorKind::Unimplemented => {
                f.write_fmt(format_args!("unimplemented {}", self.desc))?
            }
        }
        Ok(())
    }
}
