use clap::{arg, Command};
use goblin::Object;
use std::fs;
use validator::error::{Error, Result};

/// replace ".long 0x00000000" with s_nop 0 in .text section
/// This is not in the TCB

const S_NOP_0: u32 = 0xBF800000;

struct Replacer {
    bytes: Vec<u8>,
}

impl Replacer {
    fn new(file: &str) -> Result<Replacer> {
        let bytes = fs::read(file)?;
        Ok(Replacer { bytes })
    }

    fn save(&self, file: &str) -> Result<()> {
        fs::write(file, &self.bytes)?;
        Ok(())
    }

    fn process(&self) -> Result<()> {
        let elf = match Object::parse(&self.bytes)? {
            Object::Elf(x) => Ok(x),
            _ => Err(Error::ELFError(goblin::error::Error::Malformed(
                "Invalid ELF".to_string(),
            ))),
        }?;
        let text_sec = elf
            .section_headers
            .iter()
            .find(|h| {
                matches!(elf.shdr_strtab.get_at(h.sh_name),
                Some(x) if x == ".text")
            })
            .ok_or_else(|| {
                Error::ELFError(goblin::error::Error::Malformed(
                    "No .text section".to_string(),
                ))
            })?;
        let text = unsafe {
            let text_ptr = self.bytes.as_ptr().add(text_sec.sh_addr as usize) as *mut u32;
            &mut *std::ptr::slice_from_raw_parts_mut(text_ptr, (text_sec.sh_size / 4) as usize)
        };
        text.iter_mut().for_each(|x| {
            if *x == 0 {
                *x = S_NOP_0;
            }
        });
        Ok(())
    }
}

fn run(input_file: &str, output_file: &str) -> Result<()> {
    let replacer = Replacer::new(input_file)?;
    replacer.process()?;
    replacer.save(output_file)?;
    Ok(())
}

fn main() {
    let matches = Command::new("Simple utility to disassemble RDNA2 instructions")
        .arg(arg!(<INPUT>))
        .arg(arg!(<OUTPUT>));
    let args = matches.get_matches();

    let input_file = args.get_one::<String>("INPUT").unwrap();
    let output_file = args.get_one::<String>("OUTPUT").unwrap();
    if let Err(x) = run(input_file, output_file) {
        println!("Failed to process the binary: {}", x);
        std::process::exit(-1);
    }
}
