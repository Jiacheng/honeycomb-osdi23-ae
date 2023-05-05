use crate::error::Token;
use bitflags::bitflags;
use std::fmt::{Display, Formatter};

bitflags! {
    pub struct RelationFlags : u32 {
        const PURE = 1;
        const PREDICATE = 1 << 1;
        const ACTION = 1 << 2;
        const EXTERNAL = 1 << 3;
        const INTRINSIC = 1 << 4;
        // Whether the relation should be generated
        const GLOBAL = 1 << 5;

        const HAS_SINGLE_OUT_ARG = 1 << 6;
        const HAS_MULTI_OUT_ARGS = 1 << 7;
    }

    pub struct ArgumentFlag : u32 {
        const OUT = 1;
    }
}

#[derive(Debug, Clone)]
pub struct Identifier<'a> {
    pub pos: Token<'a>,
    pub id: String,
}

#[derive(Debug, Clone)]
pub struct Declaration<'a> {
    pub pos: Token<'a>,
    pub name: Identifier<'a>,
    pub flags: RelationFlags,
    pub params: Vec<Argument<'a>>,
    pub annotations: Vec<DeclAnnotation>,
}

#[derive(Debug, Clone)]
pub enum DeclAnnotation {
    CFuncName(String),
}

#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub enum Type {
    BOOL,
    U32,
    U64,
    STRING,
    VOID,
    OPAQUE(String),
}

impl Type {
    pub fn can_cast_to(&self, n: &Type) -> bool {
        self == n
            || match self {
                Type::BOOL => *n == Type::U32 || *n == Type::U64,
                Type::U32 => *n == Type::U64,
                _ => false,
            }
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::BOOL => f.write_str("bool"),
            Type::U32 => f.write_str("u32"),
            Type::U64 => f.write_str("u64"),
            Type::STRING => f.write_str("string"),
            Type::VOID => f.write_str("void"),
            Type::OPAQUE(x) => f.write_fmt(format_args!("opaque({})", x)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Argument<'a> {
    pub pos: Token<'a>,
    pub name: Identifier<'a>,
    pub flag: ArgumentFlag,
    pub ty: Type,
}

// TODO: Refactor
#[derive(Debug)]
pub struct Metadata {
    pub c_name: String,
}

#[derive(Debug, Clone)]
pub enum Term<'a> {
    IDENTIFIER(Identifier<'a>),
    STRING(String),
    INTEGER(u64),
}

#[derive(Debug, Clone)]
pub struct Atom<'a> {
    pub pos: Token<'a>,
    pub relation: Identifier<'a>,
    pub terms: Vec<Term<'a>>,
}

#[derive(Debug, Clone)]
pub struct Rule<'a> {
    pub pos: Token<'a>,
    pub head: Atom<'a>,
    pub predicates: Vec<Atom<'a>>,
}

#[derive(Debug)]
pub struct Program<'a> {
    pub declarations: Vec<Declaration<'a>>,
    pub rules: Vec<Rule<'a>>,
}
