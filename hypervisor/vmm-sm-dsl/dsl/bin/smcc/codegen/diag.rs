use crate::error::{Error, SemanticError, Token};
use std::cell::RefCell;

#[derive(Debug)]
pub struct DiagnosticContext<'a> {
    lazy_pos: RefCell<Option<Token<'a>>>,
}

impl<'a> Default for DiagnosticContext<'a> {
    fn default() -> Self {
        Self {
            lazy_pos: RefCell::new(None),
        }
    }
}

impl<'a> DiagnosticContext<'a> {
    pub fn set_location(&self, tok: &Token<'a>) {
        self.lazy_pos.replace(Some(tok.clone()));
    }

    pub fn err_with_location(&self, e: Error) -> Error {
        let l = self.lazy_pos.borrow();
        let loc = l.map(|x| x.clone().into());
        match e {
            Error::SemanticError(e) => Error::SemanticError(SemanticError {
                loc,
                kind: e.kind,
                desc: e.desc,
            }),
            _ => e,
        }
    }
}
