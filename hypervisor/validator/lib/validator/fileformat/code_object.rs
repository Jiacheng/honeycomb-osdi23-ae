use crate::error::{Error, Result};
use crate::fileformat::msgpack::{Seq, StreamParser};
use crate::fileformat::parser::{ArgInfo, KernelInfo};
use std::io;
use std::io::ErrorKind;
use std::str::FromStr;

use super::parser::KernelArgValueKind;

pub(crate) struct CodeObjectV3 {}

impl CodeObjectV3 {
    pub(crate) fn try_foreach<'a, F>(buf: &'a [u8], mut f: F) -> Result<()>
    where
        F: FnMut(&mut KernelInfo<'a>, &'a str) -> Result<()>,
    {
        let p = &mut StreamParser::new(buf);
        let (m, p) = p.read_map()?;
        m.try_foreach(p, |b| {
            let (k, b) = b.read_str()?;
            let (v, b) = b.read_any()?;
            if k == "amdhsa.kernels" {
                let arr = v
                    .get_array()
                    .ok_or_else(|| Error::IOError(io::Error::from(ErrorKind::InvalidData)))?;
                let b = arr.try_foreach(b, |b| {
                    let (m, b) = b.read_map()?;
                    let (mut ki, desc_symbol, b0) = Self::parse_kernel_info(b, &m)?;
                    (f)(&mut ki, desc_symbol)?;
                    Ok(b0)
                })?;
                Ok(b)
            } else {
                let b = v.skip_all(b)?;
                Ok(b)
            }
        })?;
        Ok(())
    }

    fn parse_kernel_args<'a, 'b>(
        p: &'b mut StreamParser<'a>,
        args: &Seq,
        arg_info: &mut Vec<ArgInfo<'a>>,
    ) -> Result<&'b mut StreamParser<'a>> {
        let no_size_err = || {
            Error::ELFError(goblin::error::Error::Malformed(
                "size required for kernel argument".to_string(),
            ))
        };
        let no_offset_err = || {
            Error::ELFError(goblin::error::Error::Malformed(
                "offset required for kernel argument".to_string(),
            ))
        };
        let no_value_kind_err = || {
            Error::ELFError(goblin::error::Error::Malformed(
                "value_kind required for kernel argument".to_string(),
            ))
        };
        args.try_foreach(p, |b| {
            let (m, b) = b.read_map()?;
            let mut name = None;
            let mut size = None;
            let mut offset = None;
            let mut value_kind = None;
            let b = m.try_foreach(b, |b| {
                let (k, b) = b.read_str()?;
                let b = match k {
                    ".name" => {
                        let (n, b) = b.read_str()?;
                        name = Some(n);
                        b
                    }
                    ".size" => {
                        let (sz, b) = b.read_integer()?;
                        let sz = TryInto::<u64>::try_into(sz)? as usize;
                        size = Some(sz);
                        b
                    }
                    ".offset" => {
                        let (off, b) = b.read_integer()?;
                        let off = TryInto::<u64>::try_into(off)? as usize;
                        offset = Some(off);
                        b
                    }
                    ".value_kind" => {
                        let (kind, b) = b.read_str()?;
                        value_kind = Some(KernelArgValueKind::from_str(kind)?);
                        b
                    }
                    _ => {
                        let (v, b) = b.read_any()?;
                        v.skip_all(b)?
                    }
                };
                Ok(b)
            })?;
            arg_info.push(ArgInfo {
                name,
                offset: offset.ok_or_else(no_offset_err)?,
                length: size.ok_or_else(no_size_err)?,
                value_kind: value_kind.ok_or_else(no_value_kind_err)?,
            });
            Ok(b)
        })
    }

    fn parse_kernel_info<'a, 'b>(
        p: &'b mut StreamParser<'a>,
        m: &Seq,
    ) -> Result<(KernelInfo<'a>, &'a str, &'b mut StreamParser<'a>)> {
        let mut ki = KernelInfo::default();
        let mut desc_symbol: &'a str = "";
        let p = m.try_foreach(p, |b| {
            let (k, b) = b.read_str()?;
            let b = match k {
                ".name" => {
                    let (v, b) = b.read_str()?;
                    ki.name = v;
                    b
                }
                ".symbol" => {
                    let (v, b) = b.read_str()?;
                    desc_symbol = v;
                    b
                }
                ".kernarg_segment_align" => {
                    let (v, b) = b.read_integer()?;
                    ki.kern_arg_segment_align = TryInto::<u64>::try_into(v)? as u32;
                    b
                }
                ".args" => {
                    let (v, b) = b.read_array()?;
                    Self::parse_kernel_args(b, &v, &mut ki.arguments)?
                }
                _ => {
                    let (v, b) = b.read_any()?;
                    v.skip_all(b)?
                }
            };
            Ok(b)
        })?;

        if !ki.name.is_empty() && !desc_symbol.is_empty() {
            Ok((ki, desc_symbol, p))
        } else {
            Err(Error::ELFError(goblin::error::Error::Malformed(
                "Kernel descriptor with no name".to_string(),
            )))
        }
    }
}
