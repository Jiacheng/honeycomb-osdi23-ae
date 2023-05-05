use crate::error::Error;
use std::cell::RefCell;
use std::fmt::{Debug, Display, Formatter};

#[derive(Copy, Clone, Debug)]
pub enum Severity {
    Info,
}

impl Display for Severity {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Info => f.pad("info"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BinaryLocation {
    pub(crate) name: String,
    pub(crate) offset: usize,
}

#[derive(Clone, Debug)]
pub enum Location {
    BinaryLocation(BinaryLocation),
    Descriptor,
}

#[derive(Debug)]
pub struct Remark {
    pub(crate) loc: Location,
    pub(crate) severity: Severity,
    pub(crate) err: Error,
    pub(crate) desc: Option<String>,
}

#[derive(Debug, Default)]
pub struct DiagnosticContext {
    remarks: RefCell<Vec<Remark>>,
}

impl<'a> Remark {
    pub fn kernel(name: &'a str, offset: usize, err: Error) -> Remark {
        Self {
            loc: Location::BinaryLocation(BinaryLocation {
                name: name.to_string(),
                offset,
            }),
            severity: Severity::Info,
            err,
            desc: None,
        }
    }

    pub fn constraints(err: Error, desc: Option<String>) -> Remark {
        Remark {
            loc: Location::Descriptor,
            severity: Severity::Info,
            err,
            desc,
        }
    }
}

impl Display for Location {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Location::BinaryLocation(b) => write!(f, "{}:{}", b.name, b.offset),
            Location::Descriptor => f.pad("<descriptor>"),
        }
    }
}

impl Display for Remark {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: [{}] ", self.loc, self.severity)?;
        Display::fmt(&self.err, f)?;
        if let Some(x) = &self.desc {
            write!(f, " {}", x)?;
        }
        Ok(())
    }
}

impl DiagnosticContext {
    pub fn record(&self, remark: Remark) {
        self.remarks.borrow_mut().push(remark);
    }

    pub fn remarks(&self) -> Vec<Remark> {
        self.remarks.take()
    }
}
