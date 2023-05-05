use crate::error::{Error, Result};
use std::io;
use std::io::ErrorKind;

#[derive(Copy, Clone)]
pub(crate) struct StreamParser<'a> {
    bytes: &'a [u8],
    current_position: usize,
}

#[derive(Ord, PartialOrd, Eq, PartialEq)]
pub(crate) enum ObjectType {
    Integer,
    String,
    Map,
    Array,
}

impl ObjectType {
    fn get_type(c: u8) -> Option<ObjectType> {
        if (c >> 7) == 0b0
            || (c >> 5) == 0b111
            || (0xc2..=0xc3).contains(&c)
            || (0xcc..=0xd3).contains(&c)
        {
            Some(ObjectType::Integer)
        } else if (c >> 5) == 0b101 || (0xd9..=0xdb).contains(&c) {
            Some(ObjectType::String)
        } else if (c >> 4) == 0b1000 || (0xde..=0xdf).contains(&c) {
            Some(ObjectType::Map)
        } else if (c >> 4) == 0b1001 || (0xdc..=0xdd).contains(&c) {
            Some(ObjectType::Array)
        } else {
            None
        }
    }
}

pub(crate) enum Object<'a> {
    Integer(Integer),
    String(&'a str),
    Map(Seq),
    Array(Seq),
}

#[derive(Copy, Clone, Debug)]
pub enum Integer {
    Unsigned(u64),
    Signed(i64),
}

impl<'a> Object<'a> {
    pub(crate) fn get_array(&self) -> Option<&Seq> {
        match self {
            Object::Array(x) => Some(x),
            _ => None,
        }
    }

    pub(crate) fn skip_all<'b>(
        &self,
        p: &'b mut StreamParser<'a>,
    ) -> Result<&'b mut StreamParser<'a>> {
        let r: &'b mut StreamParser<'a> = match self {
            Object::Integer(_) | Object::String(_) => Ok(p),
            Object::Map(x) => (0..x.num_entries * 2).try_fold(p, |p, _| {
                let (o, p) = p.read_any()?;
                o.skip_all(p)
            }),
            Object::Array(x) => (0..x.num_entries).try_fold(p, |p, _| {
                let (o, p) = p.read_any()?;
                o.skip_all(p)
            }),
        }?;
        Ok(r)
    }
}

impl TryInto<u64> for Integer {
    type Error = Error;

    fn try_into(self) -> std::result::Result<u64, Self::Error> {
        match self {
            Integer::Unsigned(x) => Ok(x),
            Integer::Signed(_) => Err(Error::IOError(io::Error::from(ErrorKind::InvalidData))),
        }
    }
}

impl TryInto<i64> for Integer {
    type Error = Error;

    fn try_into(self) -> std::result::Result<i64, Self::Error> {
        match self {
            Integer::Unsigned(_) => Err(Error::IOError(io::Error::from(ErrorKind::InvalidData))),
            Integer::Signed(x) => Ok(x),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct Seq {
    num_entries: usize,
}

impl Seq {
    pub(crate) fn try_foreach<'a, 'b, F>(
        &self,
        p: &'b mut StreamParser<'a>,
        mut f: F,
    ) -> Result<&'b mut StreamParser<'a>>
    where
        F: FnMut(&'b mut StreamParser<'a>) -> Result<&'b mut StreamParser<'a>>,
    {
        (0..self.num_entries).try_fold(p, |p, _| (f)(p))
    }
}

impl<'a> StreamParser<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> StreamParser<'a> {
        StreamParser {
            bytes,
            current_position: 0,
        }
    }

    pub(crate) fn read_any(&mut self) -> Result<(Object<'a>, &mut Self)> {
        let c = self.peek_u8()?;
        let t = ObjectType::get_type(c)
            .ok_or_else(|| Error::IOError(io::Error::from(ErrorKind::InvalidData)))?;
        match t {
            ObjectType::Integer => {
                let r = self.read_integer()?;
                Ok((Object::Integer(r.0), r.1))
            }
            ObjectType::String => {
                let r = self.read_str()?;
                Ok((Object::String(r.0), r.1))
            }
            ObjectType::Map => {
                let r = self.read_map()?;
                Ok((Object::Map(r.0), r.1))
            }
            ObjectType::Array => {
                let r = self.read_array()?;
                Ok((Object::Array(r.0), r.1))
            }
        }
    }

    pub(crate) fn read_map(&mut self) -> Result<(Seq, &mut Self)> {
        let c = self.read_u8()?;
        Self::expect_type(c, ObjectType::Map)?;
        let num_entries = if (c >> 4) == 0b1000 {
            Ok((c - 0x80) as u64)
        } else {
            self.read_be(2 << (c - 0xde))
        }? as usize;
        Ok((Seq { num_entries }, self))
    }

    pub(crate) fn read_array(&mut self) -> Result<(Seq, &mut Self)> {
        let c = self.read_u8()?;
        Self::expect_type(c, ObjectType::Array)?;
        let num_entries = if (c >> 4) == 0b1001 {
            Ok((c - 0x90) as u64)
        } else {
            self.read_be(2 << (c - 0xdc))
        }? as usize;
        Ok((Seq { num_entries }, self))
    }

    pub(crate) fn read_integer(&mut self) -> Result<(Integer, &mut Self)> {
        let c = self.read_u8()?;
        Self::expect_type(c, ObjectType::Integer)?;
        let r: Result<Integer> = if (c >> 7) == 0b0 {
            Ok(Integer::Unsigned(c as u64))
        } else if (c >> 5) == 0b111 {
            Ok(Integer::Signed(c as i8 as i64))
        } else if (0xc2..=0xc3).contains(&c) {
            Ok(Integer::Unsigned((c - 0xc2) as u64))
        } else {
            // (0xcc <= c && c <= 0xd3)
            let length = 1 << ((c - 0xcc) % 4);
            let is_unsigned = c <= 0xcf;
            let v = self.read_be(length)?;
            if is_unsigned {
                Ok(Integer::Unsigned(v))
            } else if length == 1 {
                Ok(Integer::Signed(v as u8 as i8 as i64))
            } else if length == 2 {
                Ok(Integer::Signed(v as u16 as i16 as i64))
            } else if length == 4 {
                Ok(Integer::Signed(v as u32 as i32 as i64))
            } else {
                Ok(Integer::Signed(v as i64))
            }
        };
        Ok((r?, self))
    }

    pub(crate) fn read_str(&mut self) -> Result<(&'a str, &mut Self)> {
        let c = self.read_u8()?;
        Self::expect_type(c, ObjectType::String)?;
        let length = if (c >> 5) == 0b101 {
            Ok((c & 0x1f) as u64)
        } else {
            self.read_be(1 << ((c - 0xd9) % 4))
        }? as usize;
        let slice = self.read_slice(length)?;
        let r = std::str::from_utf8(slice)
            .map_err(|_| Error::IOError(io::Error::from(ErrorKind::InvalidData)))?;
        Ok((r, self))
    }

    fn peek_u8(&mut self) -> Result<u8> {
        Ok(self.peek_slice(1)?[0])
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_slice(1)?[0])
    }

    fn read_be(&mut self, l: usize) -> Result<u64> {
        Ok(self
            .read_slice(l)?
            .iter()
            .fold(0u64, |x, y| (x << 8) | (*y) as u64))
    }

    fn peek_slice(&mut self, l: usize) -> Result<&'a [u8]> {
        if self.current_position + l > self.bytes.len() {
            Err(Error::IOError(std::io::Error::from(
                ErrorKind::UnexpectedEof,
            )))
        } else {
            let r = &self.bytes[self.current_position..self.current_position + l];
            Ok(r)
        }
    }

    fn read_slice(&mut self, l: usize) -> Result<&'a [u8]> {
        let r = self.peek_slice(l)?;
        self.current_position += l;
        Ok(r)
    }

    fn expect_type(c: u8, m: ObjectType) -> Result<()> {
        let t = ObjectType::get_type(c);
        if t.is_none() || t.unwrap() != m {
            return Err(Error::IOError(std::io::Error::from(ErrorKind::InvalidData)));
        }
        Ok(())
    }
}
