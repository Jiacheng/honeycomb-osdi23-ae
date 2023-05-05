use crate::isa::rdna2::opcodes::*;
use std::f64::consts::PI;
use std::fmt::{Display, Formatter, LowerHex};

/**
 * A representation of the operand encoding in the instruction. There are three types of encoding:
 * * Scalar instruction, 8-bit: up to SpecialRegister
 * * Vector instruction, 9-bit: everything
 * * Vector instruction, 8-bit, only VectorRegister is valid
 **/
#[derive(Copy, Clone, Debug)]
pub enum Operand {
    // RDNA2 Instruction Set Reference 13.3.6, Table 86
    // 0 - 105, SGPR0 - SGPR105
    // The second option contains the length of the register group
    ScalarRegister(u8, u8),
    // Constant. There are three types of constants:
    // * 128 - 208, constant integer: Record the actual value
    // * SIMM16 (signed immediate) in SOPP instruction
    // * 255. 32-bit literal
    Constant(i32),
    // 240 - 247, constant float. Record an index to the table to save space
    ConstantFloat(u8),
    // 106-127, 251-253. Special registers such as trap registers / VCC / EXEC.
    // Note that VCC / EXEC are aliased to the uppermost scalar registers.
    // Need to check there is no malicious use.
    SpecialScalarRegister(u8),
    // 256 - 511. VGPR0-VGPR255
    // The second option contains the length of the register group
    VectorRegister(u8, u8),
    // This is a placeholder for the destination of the store instructions.
    Void,
    Reserved,
}

impl Operand {
    pub const CONSTANT_FLOATS: [f64; 9] =
        [0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 1.0 / (2.0 * PI)];
    pub const CONSTANT_FLOATS_STR: [&'static str; 9] = [
        "0.5",
        "-0.5",
        "1.0",
        "-1.0",
        "2.0",
        "-2.0",
        "4.0",
        "-4.0",
        "0.15915494",
    ];
    pub const SPECIAL_REG_VCC_LO: u8 = 106;
    pub const SPECIAL_REG_VCC_HI: u8 = 107;
    pub const SPECIAL_REG_M0: u8 = 124;
    pub const SPECIAL_REG_NULL: u8 = 125;
    pub const SPECIAL_REG_EXEC_LO: u8 = 126;
    pub const SPECIAL_REG_EXEC_HI: u8 = 127;
    pub const SDWA: u32 = 249;
    pub const DPP16: u32 = 250;
    pub const SPECIAL_REG_VCCZ: u8 = 251;
    pub const SPECIAL_REG_EXECZ: u8 = 252;
    pub const SPECIAL_REG_SCC: u8 = 253;
    pub const SPECIAL_REG_LDS_DIRECT: u8 = 254;
    pub const SPECIAL_REG_NAMES: [(u8, &'static str); 10] = [
        (Operand::SPECIAL_REG_VCC_LO, "vcc_lo"),
        (Operand::SPECIAL_REG_VCC_HI, "vcc_hi"),
        (Operand::SPECIAL_REG_M0, "m0"),
        (Operand::SPECIAL_REG_NULL, "null"),
        (Operand::SPECIAL_REG_EXEC_LO, "exec_lo"),
        (Operand::SPECIAL_REG_EXEC_HI, "exec_hi"),
        (Operand::SPECIAL_REG_VCCZ, "vccz"),
        (Operand::SPECIAL_REG_EXECZ, "execz"),
        (Operand::SPECIAL_REG_SCC, "scc"),
        (Operand::SPECIAL_REG_LDS_DIRECT, "lds_direct"),
    ];
    /// Name of special registers used as explicit operands
    /// For now special registers with code greater than 127 will never be the *explicit* destination.
    /// `scc` would be the destination register (e.g. in `s_cmp_eq_i32`)
    /// but will not be explicitly output in the instruction.
    /// The disassembly of LLVM marks these registers with a prefix `src_`
    pub const SPECIAL_REG_OPERAND_NAME: [(u8, &'static str); 10] = [
        (Operand::SPECIAL_REG_VCC_LO, "vcc_lo"),
        (Operand::SPECIAL_REG_VCC_HI, "vcc_hi"),
        (Operand::SPECIAL_REG_M0, "m0"),
        (Operand::SPECIAL_REG_NULL, "null"),
        (Operand::SPECIAL_REG_EXEC_LO, "exec_lo"),
        (Operand::SPECIAL_REG_EXEC_HI, "exec_hi"),
        (Operand::SPECIAL_REG_VCCZ, "src_vccz"),
        (Operand::SPECIAL_REG_EXECZ, "src_execz"),
        (Operand::SPECIAL_REG_SCC, "src_scc"),
        (Operand::SPECIAL_REG_LDS_DIRECT, "src_lds_direct"),
    ];
    pub const MAX_SGPR_NUM: usize = 105;

    pub fn is_special_register(&self, id: u8) -> bool {
        matches!(self, Self::SpecialScalarRegister(x) if *x == id)
    }
}

/// Display method for Operand uses the alternate flag and precision of the Formatter
/// - alternate ("{:#}"): format constant as hexadecimal if it is large, decimal otherwise
/// - precision ("{:.*}"): the size of the register group
/// Use LowerHex with alternate flag ("{:#x}") to format constant as hexadecimal
impl Display for Operand {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Operand::ScalarRegister(r, len) => {
                assert_eq!(*len as usize, f.precision().unwrap_or(1));
                Operand::fmt_register(f, 's', *r, f.precision().unwrap_or(1))
            }
            Operand::Constant(v) => {
                if f.alternate() && !(-16..=64).contains(v) {
                    write!(f, "{:#x}", *v)
                } else {
                    write!(f, "{}", *v)
                }
            }
            Operand::ConstantFloat(idx) => {
                write!(f, "{}", Operand::CONSTANT_FLOATS_STR[*idx as usize])
            }
            Operand::SpecialScalarRegister(idx) => {
                let size = f.precision().unwrap_or(1);
                match (size, *idx) {
                    (1, _) => {
                        let name = Operand::SPECIAL_REG_OPERAND_NAME
                            .iter()
                            .find(|(id, _)| idx == id)
                            .expect("Unknown special register");
                        f.write_str(name.1)
                    }
                    (2, Operand::SPECIAL_REG_EXEC_LO) => f.write_str("exec"),
                    (2, Operand::SPECIAL_REG_VCC_LO) => f.write_str("vcc"),
                    _ => Err(std::fmt::Error),
                }
            }
            Operand::VectorRegister(r, len) => {
                assert_eq!(*len as usize, f.precision().unwrap_or(1));
                Operand::fmt_register(f, 'v', *r, f.precision().unwrap_or(1))
            }
            Operand::Void => f.pad("void"),
            Operand::Reserved => f.pad("reserved"),
        }
    }
}

impl LowerHex for Operand {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Operand::Constant(v) => {
                if *v < 0 {
                    f.pad("-")?;
                    LowerHex::fmt(&v.unsigned_abs(), f)
                } else {
                    LowerHex::fmt(v, f)
                }
            }
            _ => Err(std::fmt::Error),
        }
    }
}

impl Operand {
    fn fmt_register(
        f: &mut Formatter<'_>,
        prefix: char,
        idx: u8,
        width: usize,
    ) -> std::fmt::Result {
        if width == 1 {
            write!(f, "{}{}", prefix, idx)
        } else {
            write!(f, "{}[{}:{}]", prefix, idx, (idx as usize) + width - 1)
        }
    }
}

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Debug)]
#[allow(non_camel_case_types)]
pub enum Opcode {
    SOP2(SOP2Opcode),
    SOPK(SOPKOpcode),
    SOP1(SOP1Opcode),
    SOPC(SOPCOpcode),
    SOPP(SOPPOpcode),
    SMEM(SMEMOpcode),
    VOP2(VOP2Opcode),
    VOP1(VOP1Opcode),
    VOPC(VOPCOpcode),
    VOP3P(VOP3POpcode),
    VINTERP,
    LDSGDS(LDSGDSOpcode),
    VOP3AB(VOP3ABOpcode),
    MUBUF(MUBUFOpcode),
    MTBUF,
    MIMG,
    EXPORT,
    VMEM(VMEMOpcode),
    // A phantom opcode that marks invalid instructions
    INVALID(u32),
}

impl Opcode {
    /**
     * The number of dwords (uint) that encodes the opcode.
     * Note that there might be an additional one dword if one of the operands uses literal constants.
     **/
    pub fn get_opcode_dwords(&self) -> usize {
        use Opcode::*;
        match self {
            SMEM(_) | VOP3AB(_) | VOP3P(_) | LDSGDS(_) | MUBUF(_) | MTBUF | EXPORT | VMEM(_) => 2,
            MIMG => 5,
            _ => 1,
        }
    }

    pub fn get_num_operands(&self) -> usize {
        use Opcode::*;
        match self {
            SOP2(_) | VOP2(_) => 3,
            SOPK(_) | SOP1(_) | SOPC(_) | VOP1(_) | VOPC(_) => 2,
            SOPP(_) => 1,
            SMEM(_) => 3,
            VINTERP => 3,
            LDSGDS(_) => 5,
            VOP3AB(_) | VOP3P(_) => 4,
            MUBUF(_) | MTBUF => 6,
            MIMG => 16,
            EXPORT => 5,
            VMEM(_) => 6,
            _ => unreachable!(),
        }
    }
}

impl Opcode {
    fn try_parse_opcode(op: u32) -> Option<Opcode> {
        // Longest prefix takes precedences
        if op >> 23 == 0x17du32 {
            Some(Opcode::SOP1(
                SOP1Opcode::try_from((op >> 8) & 0xffu32).ok()?,
            ))
        } else if op >> 23 == 0x17eu32 {
            Some(Opcode::SOPC(
                SOPCOpcode::try_from((op >> 16) & 0x7fu32).ok()?,
            ))
        } else if op >> 23 == 0x17fu32 {
            Some(Opcode::SOPP(
                SOPPOpcode::try_from((op >> 16) & 0x7fu32).ok()?,
            ))
        } else if op >> 25 == 0x3fu32 {
            Some(Opcode::VOP1(
                VOP1Opcode::try_from((op >> 9) & 0xffu32).ok()?,
            ))
        } else if op >> 25 == 0x3eu32 {
            Some(Opcode::VOPC(
                VOPCOpcode::try_from((op >> 17) & 0xffu32).ok()?,
            ))
        } else if op >> 26 == 0x33u32 {
            Some(Opcode::VOP3P(
                VOP3POpcode::try_from((op >> 16) & 0x7fu32).ok()?,
            ))
        } else if op >> 26 == 0x35u32 {
            Some(Opcode::VOP3AB(
                VOP3ABOpcode::try_from((op >> 16) & 0x3ffu32).ok()?,
            ))
        } else if op >> 26 == 0x36u32 {
            Some(Opcode::LDSGDS(
                LDSGDSOpcode::try_from((op >> 18) & 0xffu32).ok()?,
            ))
        } else if op >> 26 == 0x37u32 {
            Some(Opcode::VMEM(
                VMEMOpcode::try_from((op >> 18) & 0x7fu32).ok()?,
            ))
        } else if op >> 26 == 0x38u32 {
            Some(Opcode::MUBUF(
                MUBUFOpcode::try_from((op >> 18) & 0x7fu32).ok()?,
            ))
        } else if op >> 26 == 0x3du32 {
            Some(Opcode::SMEM(
                SMEMOpcode::try_from((op >> 18) & 0xffu32).ok()?,
            ))
        } else if op >> 28 == 0xbu32 {
            Some(Opcode::SOPK(
                SOPKOpcode::try_from((op >> 23) & 0x1fu32).ok()?,
            ))
        } else if op >> 30 == 2u32 {
            Some(Opcode::SOP2(
                SOP2Opcode::try_from((op >> 23) & 0x7fu32).ok()?,
            ))
        } else if op >> 31 == 0u32 {
            Some(Opcode::VOP2(
                VOP2Opcode::try_from((op >> 25) & 0x7fu32).ok()?,
            ))
        } else {
            None
        }
    }
}

impl From<u32> for Opcode {
    fn from(op: u32) -> Self {
        Opcode::try_parse_opcode(op).unwrap_or(Opcode::INVALID(op))
    }
}

impl Display for Opcode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use Opcode::*;
        match self {
            SOP2(x) => x.fmt(f),
            SOP1(x) => x.fmt(f),
            SOPK(x) => x.fmt(f),
            SOPC(x) => x.fmt(f),
            SOPP(x) => x.fmt(f),
            SMEM(x) => x.fmt(f),
            VOP2(x) => x.fmt(f),
            VOP1(x) => x.fmt(f),
            VOPC(x) => x.fmt(f),
            VOP3AB(x) => x.fmt(f),
            VOP3P(x) => x.fmt(f),
            VMEM(x) => x.fmt(f),
            LDSGDS(x) => x.fmt(f),
            MUBUF(x) => x.fmt(f),
            INVALID(_) => f.pad("invalid"),
            _ => todo!(),
        }
    }
}
