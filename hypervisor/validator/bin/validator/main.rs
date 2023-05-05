extern crate json;
mod descriptor;

use crate::descriptor::Descriptor;
use clap::{arg, Command};
use std::collections::HashMap;
use std::fs;
use validator::error::Result;
use validator::fileformat;
use validator::ir::RuntimeInfo;
use validator::prover::SymbolicHeap;
use validator::support::diagnostic::DiagnosticContext;

struct Validator {}

impl Validator {
    fn run(file: &str, desc_file: &str, diag: &DiagnosticContext) -> Result<()> {
        let descriptors = Descriptor::load(desc_file, diag)?;
        let data = fs::read(file)?;
        let modules = fileformat::disassemble(data.as_slice())?;

        for (ki, f) in modules.kernels {
            if let Some(desc) = descriptors.get(ki.name) {
                let mut heap = SymbolicHeap::new();
                let arguments = ki
                    .arguments
                    .iter()
                    .map(|arg_info| (arg_info.offset, arg_info))
                    .collect::<HashMap<usize, &fileformat::ArgInfo>>();
                desc.constraints
                    .iter()
                    .try_for_each(|c| heap.load_constraints(c, &arguments))?;
                let rt_info = RuntimeInfo::new(desc.block_size, desc.grid_size);
                validator::prover::prove(&ki, &f, &mut heap, &rt_info, diag, &desc.runtime_checks);
            }
        }
        Ok(())
    }
}

pub fn main() {
    let matches = Command::new("Validator")
        .arg(arg!(<bin> "The ELF binary"))
        .arg(arg!(<desc> "The JSON descriptor of the constraints"))
        .get_matches();

    let input_file = matches.get_one::<String>("bin").unwrap();
    let constraints = matches.get_one::<String>("desc").unwrap();
    let diag = DiagnosticContext::default();

    let mut ret = 0;
    if let Err(x) = Validator::run(input_file.as_str(), constraints.as_str(), &diag) {
        println!("Failed to load the binary: {}", x);
        ret = -1;
    }

    let r = diag.remarks();
    r.iter().for_each(|r| println!("{}", r));
    println!("Generates {} remarks", r.len());
    std::process::exit(ret);
}
