use json::JsonValue;
use std::collections::{HashMap, HashSet};
use std::{fs, io};
use validator::error::{Error, Result};
use validator::ir::constraints::{Constraint, Location, MemoryConstraint};
use validator::support::diagnostic::{DiagnosticContext, Remark};

#[derive(Debug, Clone)]
pub(crate) struct Descriptor {
    pub(crate) constraints: Vec<MemoryConstraint>,
    pub(crate) block_size: [usize; 3],
    pub(crate) grid_size: [usize; 3],
    // The indices of memory access instructions that will be checked in runtime
    // The prover will skip them for now
    pub(crate) runtime_checks: HashSet<usize>,
}

impl Descriptor {
    pub(crate) fn load(
        file: &str,
        diag: &DiagnosticContext,
    ) -> Result<HashMap<String, Descriptor>> {
        let data = fs::read_to_string(file)?;
        let d = json::parse(data.as_str()).map_err(|json_err| {
            diag.record(Remark::constraints(
                Self::invalid_json_data(),
                Some(json_err.to_string()),
            ));
            Self::invalid_json_data()
        })?;
        let mut ret = HashMap::new();
        if let JsonValue::Array(kernels) = &d["kernels"] {
            for k in kernels {
                let name = k["name"]
                    .as_str()
                    .ok_or_else(|| Error::IOError(io::Error::from(io::ErrorKind::InvalidData)))?;
                let mut constraints = Vec::new();
                let mut block_size = [1; 3];
                let mut grid_size = [1; 3];
                let marks = if let JsonValue::Array(marks) = &k["marks"] {
                    marks
                        .iter()
                        .map(|v| v.as_usize().unwrap_or(usize::MAX))
                        .collect::<HashSet<usize>>()
                } else {
                    HashSet::<usize>::new()
                };
                if let JsonValue::Array(v) = &k["constraints"] {
                    for c in v {
                        match c["type"].as_str().ok_or_else(Self::invalid_json_data)? {
                            "argument" => constraints.push(Self::parse_kernel_arguments(c)?),
                            "block_size" => block_size = Self::parse_dimensions(c)?,
                            "grid_size" => grid_size = Self::parse_dimensions(c)?,
                            _ => {
                                return Err(Error::IOError(io::Error::from(
                                    io::ErrorKind::InvalidData,
                                )))
                            }
                        };
                    }
                    ret.insert(
                        name.to_string(),
                        Descriptor {
                            constraints,
                            block_size,
                            grid_size,
                            runtime_checks: marks,
                        },
                    );
                }
            }
        }
        Ok(ret)
    }

    fn parse_kernel_arguments(v: &JsonValue) -> Result<MemoryConstraint> {
        let c = Constraint {
            min: v["min"].as_isize().unwrap_or(0),
            max: v["max"].as_isize().ok_or_else(Self::invalid_json_data)?,
        };
        Ok(MemoryConstraint::new(
            Location::KernelArgumentPointer(
                v["offset"].as_usize().ok_or_else(Self::invalid_json_data)?,
            ),
            c,
        ))
    }

    fn parse_dimensions(v: &JsonValue) -> Result<[usize; 3]> {
        let err = Self::invalid_json_data();
        match &v["value"] {
            JsonValue::Array(dims) if dims.len() <= 3 => {
                Ok([0, 1, 2].map(|i| dims.get(i).and_then(|v| v.as_usize()).unwrap_or(1)))
            }
            _ => Err(err),
        }
    }

    #[inline]
    fn invalid_json_data() -> Error {
        Error::IOError(io::Error::from(io::ErrorKind::InvalidData))
    }
}
