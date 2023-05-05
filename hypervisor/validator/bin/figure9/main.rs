extern crate json;
mod descriptor;

use crate::descriptor::Descriptor;
use clap::{arg, Command};
use std::collections::HashMap;
use std::time::Instant;
use std::{fs, io};
use validator::error::{Error, Result};
use validator::fileformat;
use validator::ir::RuntimeInfo;
use validator::prover::SymbolicHeap;
use validator::support::diagnostic::DiagnosticContext;

struct Validator {}

impl Validator {
    fn run(file: &str, desc_file: &str, kernel: &str, diag: &DiagnosticContext) -> Result<()> {
        let descriptors = Descriptor::load(desc_file, diag)?;
        let data = fs::read(file)?;
        let (ki, f) = fileformat::disassemble_kernel(data.as_slice(), kernel)?;

        let desc = descriptors
            .get(ki.name)
            .ok_or_else(|| Error::IOError(io::Error::from(io::ErrorKind::NotFound)))?;
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
        Ok(())
    }

    fn count_inst(file: &str, func_name: &str) -> Result<usize> {
        let data = fs::read(file)?;
        let modules = fileformat::disassemble(data.as_slice())?;

        for (_, f) in modules.kernels {
            if func_name == f.name {
                return Ok(f.instructions.len());
            }
        }
        Err(Error::IOError(io::Error::from(io::ErrorKind::NotFound)))
    }
}

pub fn main() {
    let matches = Command::new("Validator")
        .arg(arg!(<bin> "The ELF binary"))
        .arg(arg!(<desc> "The JSON descriptor of the constraints"))
        .arg(arg!(<kernel> "The name of kernel to validate"))
        .get_matches();

    let input_file = matches.get_one::<String>("bin").unwrap();
    let constraints = matches.get_one::<String>("desc").unwrap();
    let kernel = matches.get_one::<String>("kernel").unwrap();
    let diag = DiagnosticContext::default();

    let mut ret = 0;
    let now = Instant::now();
    for _ in 0..100 {
        if let Err(x) = Validator::run(
            input_file.as_str(),
            constraints.as_str(),
            kernel.as_str(),
            &diag,
        ) {
            println!("Failed to load the binary: {}", x);
            ret = -1;
        }
    }
    let elapsed_time = now.elapsed();
    if let Ok(inst_count) = Validator::count_inst(input_file.as_str(), kernel.as_str()) {
        println!(
            "{}, {}",
            inst_count,
            elapsed_time.as_secs_f64() * 1000.0 / 100.0,
        );
    }
    std::process::exit(ret);
}
