use std::fs;
use validator::error::Result;
use validator::fileformat::AMDGPUELF;

struct ReadELF {
    data: Vec<u8>,
}

impl ReadELF {
    fn new(file: &str) -> Result<ReadELF> {
        let data = fs::read(file)?;
        Ok(ReadELF { data })
    }

    fn run(&self) -> Result<()> {
        let ff = AMDGPUELF::parse(self.data.as_slice())?;
        for k in ff.kernels() {
            println!("Kernel name: {}", k.name);
        }
        Ok(())
    }
}

fn run() -> Result<()> {
    let file = std::env::args().nth(1).unwrap();
    let elf = ReadELF::new(file.as_str())?;
    elf.run()
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
