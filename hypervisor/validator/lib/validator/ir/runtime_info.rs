use crate::isa::rdna2::RDNA2Target;

/**
 * Describe the runtime constraints (e.g. grid size and block size) when validating a function
 */
#[derive(Copy, Clone, Debug)]
pub struct RuntimeInfo {
    pub(crate) implicit_loop_info: ImplicitLoopInfo,
}

impl RuntimeInfo {
    pub const fn new(block_size: [usize; 3], grid_size: [usize; 3]) -> RuntimeInfo {
        RuntimeInfo {
            implicit_loop_info: ImplicitLoopInfo::new(block_size, grid_size),
        }
    }
}

/**
 * A struct to record the information of implicit loop. The validator augments the results of
 * loop bound analysis and scalar evolution to enable reasoning about relative indices of tid / gid.
 * There are six implicit loops in total w.r.t to all three dimensions, each of which has two loops for
 * the grid and the block.
 *
 * Notes:
 * (1) A typical GPU executes the loops in the order of the Z / Y / X dimensions. The current implementation
 * follows the implementation details, although the specification requires the orders to be non-deterministic.
 * (2) Currently it requires passing in the grid size of as a constant. The grid size is actually available
 * in the dispatch packet. Need to connect them together.
 */
#[derive(Copy, Clone, Debug)]
pub struct ImplicitLoopInfo {
    // Dimensions of X / Y / Z
    pub(crate) dimensions: [WorkDimension; 3],
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct WorkDimension {
    pub(crate) workgroup_size: usize,
    pub(crate) grid_size: usize,
}

impl ImplicitLoopInfo {
    pub(crate) const IMPLICIT_LOOP_NUM: usize = 6;

    // Argument ID -> implicit loop ID
    pub(crate) const ARGUMENT_DESCRIPTORS: [(usize, usize); 6] = [
        (RDNA2Target::TAG_SGPR_WORKGROUP_ID_Z, 0),
        (RDNA2Target::TAG_SGPR_TID + 2, 1),
        (RDNA2Target::TAG_SGPR_WORKGROUP_ID_Y, 2),
        (RDNA2Target::TAG_SGPR_TID + 1, 3),
        (RDNA2Target::TAG_SGPR_WORKGROUP_ID_X, 4),
        (RDNA2Target::TAG_SGPR_TID, 5),
    ];

    pub(crate) const fn new(block_size: [usize; 3], grid_size: [usize; 3]) -> ImplicitLoopInfo {
        ImplicitLoopInfo {
            dimensions: [
                WorkDimension {
                    workgroup_size: block_size[0],
                    grid_size: grid_size[0],
                },
                WorkDimension {
                    workgroup_size: block_size[1],
                    grid_size: grid_size[1],
                },
                WorkDimension {
                    workgroup_size: block_size[2],
                    grid_size: grid_size[2],
                },
            ],
        }
    }
}
