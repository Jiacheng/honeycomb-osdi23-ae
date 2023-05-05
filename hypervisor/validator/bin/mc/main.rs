use clap::{arg, Arg, Command};
use std::fs;
use std::slice::from_raw_parts;
use validator::error::Result;
use validator::isa::rdna2::Decoder;

fn run(file: &str) -> Result<()> {
    let data = fs::read(file)?;
    let d = unsafe { from_raw_parts(data.as_ptr() as *const u32, data.len() / 4) };
    let stream = Decoder::new(d);
    for (_, i) in stream {
        println!("{} ", i);
    }
    Ok(())
}

fn main() {
    let matches = Command::new("Simple utility to disassemble RDNA2 instructions")
        .arg(Arg::new("disassemble").short('d').takes_value(false))
        .arg(arg!(<BINARY>));
    let args = matches.get_matches();

    let file = args.get_one::<String>("BINARY").unwrap();

    if let Err(x) = run(file.as_str()) {
        println!("Failed to disassemble: {}", x);
        std::process::exit(-1);
    }
}
