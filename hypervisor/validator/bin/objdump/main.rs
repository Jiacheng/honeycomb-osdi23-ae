use std::fs;
use std::io;
use std::io::ErrorKind;
use validator::error::{Error, Result};
use validator::fileformat::AMDGPUELF;
use validator::isa::rdna2::Decoder;

struct ELFDumper {
    data: Vec<u8>,
}

impl ELFDumper {
    fn new(file: &str) -> Result<ELFDumper> {
        let data = fs::read(file)?;
        Ok(ELFDumper { data })
    }

    fn run(&self) -> Result<()> {
        let ff = AMDGPUELF::parse(self.data.as_slice())?;
        ff.kernels().iter().for_each(|k| {
            println!("{}:", k.name);
            Decoder::new(k.code).for_each(|(_, i)| {
                println!("\t{}", i);
            })
        });
        Ok(())
    }
}

fn run() -> Result<()> {
    let file = std::env::args()
        .nth(1)
        .ok_or_else(|| Error::IOError(io::Error::from(ErrorKind::InvalidInput)))?;
    let dumper = ELFDumper::new(file.as_str())?;
    dumper.run()
}

fn main() {
    if std::env::args().len() != 2 {
        println!("Usage: {} <input binary>", std::env::args().next().unwrap());
        std::process::exit(-1);
    }
    if let Err(x) = run() {
        println!("Failed to parse the binary: {}", x);
        std::process::exit(-1);
    }
}
