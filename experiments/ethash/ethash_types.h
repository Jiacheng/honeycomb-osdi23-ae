#pragma once

enum {
    kEthashMixBytes = 128,
    kEthashHashBytes = 64,
    kEthashHeaderBytes = 32,

    kEthashBlockSizeDag = 128,
    kEthashGridSizeDag = 8192,

    // A power of 2 here is better for performance
    kEthashDualSolverSearchResults = 4,
};

#pragma pack(push, 1)
struct EthashSolutionsOnDevice {
    unsigned count;
    // Count the wraps that are active
    unsigned hash_count;
    unsigned dummy[2];
    unsigned gid[kEthashDualSolverSearchResults];
};
#pragma pack(pop)

