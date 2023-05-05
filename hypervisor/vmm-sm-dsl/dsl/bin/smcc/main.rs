mod codegen;
mod error;

use std::cell::RefCell;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::{fmt, fs};

use clap::Parser;

use crate::codegen::ast::Program;
use crate::codegen::{lkm, parser};
use crate::error::{Error, Result, Span};
use crate::lkm::LKMCodeGen;

/// Simple program to greet a person
#[derive(Parser, Clone, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(short = 'o')]
    output: String,

    #[clap(last = true)]
    input: String,
}

struct Printer {
    pub out: BufWriter<File>,
}

impl fmt::Write for Printer {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.out
            .write(s.as_bytes())
            .map(|_| ())
            .map_err(|_| std::fmt::Error)
    }
}

impl Printer {
    fn flush(&mut self) -> Result<()> {
        self.out.flush()?;
        Ok(())
    }
}

struct Driver {
    args: Args,
}

impl Driver {
    fn create_printer(&self) -> Result<Printer> {
        let file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(self.args.output.as_str())?;
        Ok(Printer {
            out: BufWriter::new(file),
        })
    }

    fn codegen(&self, prog: Program, out: &RefCell<Printer>) -> Result<()> {
        let mut codegen = LKMCodeGen::new(out, prog);
        codegen.compile()?;
        out.borrow_mut().flush()?;
        Ok(())
    }

    fn run(&mut self) -> Result<()> {
        let src = fs::read_to_string(self.args.input.as_str())?;
        let prog = parser::program(Span::new_extra(src.as_str(), self.args.input.as_str()))
            .map_err(|e| Error::SyntaxError(format!("{}", e)))?;
        if prog.0.len() != 0 {
            return Err(Error::SyntaxError(prog.0.to_string()));
        }
        let out = RefCell::new(self.create_printer()?);
        self.codegen(prog.1, &out)
    }
}

fn main() {
    let args = Args::parse();
    let mut driver = Driver { args };
    let r = driver.run();
    if r.is_err() {
        eprintln!("Failed to compile the program: {:#?}\n", r.err().unwrap());
    }
}
