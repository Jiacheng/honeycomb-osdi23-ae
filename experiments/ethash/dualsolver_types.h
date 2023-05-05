#pragma once

#include "ethash_types.h"

enum {
    kEthashBlockSize = 64,
    kEthashGridSize = 8192,
    kEthashHashParallelism = 4,

    // A power of 2 here is better for performance
    kEthashDualSolverMaxResults = 4,
};

