use crate::codegen::ast::{ArgumentFlag, RelationFlags, Type};
use crate::error::{Error, Result, SemanticError};
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::ErrorKind;
use std::rc::Rc;
use std::{io, mem};

#[derive(Debug, Clone)]
pub struct Argument {
    pub name: String,
    pub flag: ArgumentFlag,
    pub ty: Type,
}

#[derive(Debug)]
pub struct Metadata {
    pub c_name: String,
}

#[derive(Debug)]
pub struct Relation {
    pub name: String,
    pub flag: RelationFlags,
    pub args: Vec<Argument>,
    pub return_ty: Type,
    pub md: Option<Metadata>,

    // Only support a single output argument for now
    pub out_arg_idx: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct VarExpr {
    // The real slot
    pub name: String,
    pub ty: Type,
}

impl VarExpr {
    fn new(name: &str, ty: Type) -> VarExpr {
        VarExpr {
            name: name.to_string(),
            ty,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Symbol {
    RELATION(Rc<Relation>),
    VARIABLE(VarExpr),
    FORMAL(VarExpr),
}

impl Symbol {
    pub fn get_name(&self) -> String {
        match self {
            Symbol::RELATION(x) => x.name.clone(),
            Symbol::VARIABLE(x) | Symbol::FORMAL(x) => x.name.clone(),
        }
    }

    pub fn as_relation(&self) -> Option<Rc<Relation>> {
        match self {
            Symbol::RELATION(x) => Some(x.clone()),
            _ => None,
        }
    }

    pub fn as_varexpr(&self) -> Option<VarExpr> {
        match self {
            Symbol::VARIABLE(x) | Symbol::FORMAL(x) => Some(x.clone()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
struct SymbolContext {
    symbols: HashMap<String, Symbol>,
}

#[derive(Debug, Clone)]
pub struct SymbolTable {
    contexts: RefCell<Vec<SymbolContext>>,
}

impl SymbolTable {
    pub fn new() -> SymbolTable {
        SymbolTable {
            contexts: RefCell::new(vec![]),
        }
    }

    pub fn lookup(&self, name: &str) -> Option<Symbol> {
        let c = self.contexts.borrow();
        c.iter()
            .rev()
            .find_map(|x| x.symbols.get(name).map(|x| x.clone()))
    }

    pub fn add_unique_variable(&self, name: &str, ty: Type) -> Result<String> {
        let mut i = 0;
        let mut n = name.to_string();
        while self.lookup(n.as_str()).is_some() {
            n = format!("{}_{}", name, i);
            i += 1;
        }

        let mut c = self.contexts.borrow_mut();
        let peek = c.last_mut().unwrap();
        let s = Symbol::VARIABLE(VarExpr::new(n.as_str(), ty));
        peek.symbols.insert(name.to_string(), s);
        Ok(n)
    }

    pub fn add_formal_arg(&self, arg: &Argument) -> Result<()> {
        let s = Symbol::FORMAL(VarExpr::new(arg.name.as_str(), arg.ty.clone()));
        self.try_insert_symbol(&s)
            .map(|_| ())
            .ok_or(SemanticError::redefinition(arg.name.clone()))
    }

    pub fn add_relation_decl(&self, relation: Rc<Relation>) -> Result<()> {
        let s = Symbol::RELATION(relation);
        self.try_insert_symbol(&s)
            .map(|_| ())
            .ok_or(Error::IoError(io::Error::from(ErrorKind::OutOfMemory)))
    }

    fn try_insert_symbol(&self, symbol: &Symbol) -> Option<String> {
        let m = self.lookup(symbol.get_name().as_str());
        let mut c = self.contexts.borrow_mut();
        let peek = c.last_mut().unwrap();
        if m.is_none() {
            let name = symbol.get_name();
            peek.symbols.insert(name.to_string(), symbol.clone());
            return Some(name.to_string());
        }
        None
    }

    pub fn bind(&self, name: &str, symbol: Symbol) {
        let mut c = self.contexts.borrow_mut();
        let peek = c.last_mut().unwrap();
        peek.symbols.insert(name.to_string(), symbol);
    }

    pub fn enter(&self) {
        let mut c = self.contexts.borrow_mut();
        c.push(SymbolContext {
            symbols: Default::default(),
        })
    }

    pub fn leave(&self) {
        let mut c = self.contexts.borrow_mut();
        c.pop();
    }
}

pub struct RelationBuilder {
    name: String,
    flag: RelationFlags,
    ret_ty: Type,
    args: Vec<Argument>,
    md: Option<Metadata>,
}

impl RelationBuilder {
    pub fn new(name: &str, flag: RelationFlags, ret_ty: Type) -> RelationBuilder {
        RelationBuilder {
            name: name.to_string(),
            flag,
            ret_ty,
            args: vec![],
            md: None,
        }
    }

    pub fn arg_with_flags(&mut self, name: &str, ty: Type, flag: ArgumentFlag) -> &mut Self {
        self.args.push(Argument {
            name: name.to_string(),
            flag,
            ty,
        });
        self
    }

    pub fn metadata(&mut self, md: Metadata) -> &mut Self {
        self.md = Some(md);
        self
    }

    pub fn build(&mut self) -> Relation {
        let name = self.name.clone();
        let args = mem::replace(&mut self.args, vec![]);
        let md = mem::replace(&mut self.md, None);
        let out_arg_idx: Vec<usize> = args
            .iter()
            .enumerate()
            .filter_map(|(idx, x)| {
                if x.flag.contains(ArgumentFlag::OUT) {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();
        assert!(out_arg_idx.len() <= 1);
        Relation {
            name,
            flag: self.flag,
            args,
            return_ty: self.ret_ty.clone(),
            md,
            out_arg_idx: out_arg_idx.first().cloned(),
        }
    }
}
