use crate::error::Result;
use crate::fileformat::{KernelInfo, SGPRSetup, PGMRSRC2};
use crate::ir::machine::Register;
use crate::ir::Value;

pub struct RDNA2Target {}

impl RDNA2Target {
    pub const TAG_SGPR_PRIVATE_SEGMENT_BUFFER: usize = 0;
    pub const TAG_SGPR_DISPATCH_PTR: usize = 4;
    pub const TAG_SGPR_QUEUE_PTR: usize = 6;
    pub const TAG_SGPR_KERNARG_SEGMENT_PTR: usize = 8;
    pub const TAG_SGPR_DISPATCH_ID: usize = 10;
    pub const TAG_SGPR_FLAT_SCRATCH_INIT: usize = 12;
    pub const TAG_SGPR_PRIVATE_SEGMENT_SIZE: usize = 13;
    pub const TAG_SGPR_PRIVATE_SEGMENT: usize = 14;
    pub const TAG_SGPR_WORKGROUP_ID_X: usize = 15;
    pub const TAG_SGPR_WORKGROUP_ID_Y: usize = 16;
    pub const TAG_SGPR_WORKGROUP_ID_Z: usize = 17;
    pub const TAG_SGPR_TID: usize = 18;

    /**
     * Initial kernel state set up based on https://llvm.org/docs/AMDGPUUsage.html#initial-kernel-execution-state
     **/
    pub fn get_entry_live_entries(ki: &KernelInfo) -> Result<Vec<(Register, Value)>> {
        const SGPR_REGISTERS: &[(SGPRSetup, usize, usize)] = &[
            (
                SGPRSetup::SGPR_PRIVATE_SEGMENT_BUFFER,
                4,
                RDNA2Target::TAG_SGPR_PRIVATE_SEGMENT_BUFFER,
            ),
            (
                SGPRSetup::SGPR_DISPATCH_PTR,
                2,
                RDNA2Target::TAG_SGPR_DISPATCH_PTR,
            ),
            (
                SGPRSetup::SGPR_QUEUE_PTR,
                2,
                RDNA2Target::TAG_SGPR_QUEUE_PTR,
            ),
            (
                SGPRSetup::SGPR_KERNARG_SEGMENT_PTR,
                2,
                RDNA2Target::TAG_SGPR_KERNARG_SEGMENT_PTR,
            ),
            (
                SGPRSetup::SGPR_DISPATCH_ID,
                2,
                RDNA2Target::TAG_SGPR_DISPATCH_ID,
            ),
            (
                SGPRSetup::SGPR_FLAT_SCRATCH_INIT,
                2,
                RDNA2Target::TAG_SGPR_FLAT_SCRATCH_INIT,
            ),
            (
                SGPRSetup::SGPR_PRIVATE_SEGMENT_SIZE,
                1,
                RDNA2Target::TAG_SGPR_PRIVATE_SEGMENT_SIZE,
            ),
        ];

        const SGPR_PGMRSRC2: &[(PGMRSRC2, usize, usize)] = &[
            (
                PGMRSRC2::SGPR_PRIVATE_SEGMENT,
                1,
                RDNA2Target::TAG_SGPR_PRIVATE_SEGMENT,
            ),
            (
                PGMRSRC2::SGPR_WORKGROUP_ID_X,
                1,
                RDNA2Target::TAG_SGPR_WORKGROUP_ID_X,
            ),
            (
                PGMRSRC2::SGPR_WORKGROUP_ID_Y,
                1,
                RDNA2Target::TAG_SGPR_WORKGROUP_ID_Y,
            ),
            (
                PGMRSRC2::SGPR_WORKGROUP_ID_Z,
                1,
                RDNA2Target::TAG_SGPR_WORKGROUP_ID_Z,
            ),
        ];

        let mut out = Vec::new();
        let pgmrsrc2 = PGMRSRC2::from(ki.pgmrsrcs[1]);
        SGPR_REGISTERS
            .iter()
            .filter_map(|(flag, len, tag_start)| {
                ki.setup.contains(*flag).then_some((len, tag_start))
            })
            .chain(SGPR_PGMRSRC2.iter().filter_map(|(flag, len, tag_start)| {
                pgmrsrc2.contains(*flag).then_some((len, tag_start))
            }))
            .fold(0, |start, (len, tag_start)| {
                out.extend((0..*len).map(|x| {
                    (
                        Register::Scalar((start + x) as u8),
                        Value::Argument(tag_start + x),
                    )
                }));
                start + *len
            });

        out.extend((0..=pgmrsrc2.enable_vgpr_workitem_id()?).map(|x| {
            (
                Register::Vector(x),
                Value::Argument(Self::TAG_SGPR_TID + x as usize),
            )
        }));
        Ok(out)
    }
}
