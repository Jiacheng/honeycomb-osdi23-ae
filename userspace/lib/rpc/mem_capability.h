#pragma once

#include "siphash.h"

namespace gpumpc {
namespace rpc {

__device__ static inline void ToCapability(unsigned long epoch, unsigned src,
                                           unsigned offset, unsigned length,
                                           unsigned res[8]) {
    siphash_keys keys = {
        .k0 = epoch,
        .k1 = ((unsigned long)offset << 32) | src,
        .k2 = (0x0adb042aull << 32) | length,
        .k3 = 0x5b13db8630a80964ull,
    };
    diphash_state ds(keys);
    ds.hash24();
    unsigned long mac = ds.xor_lanes();
    res[0] = epoch;
    res[1] = epoch >> 32;
    res[2] = src;
    res[3] = offset;
    res[4] = length;
    res[5] = 0;
    res[6] = mac;
    res[7] = mac >> 32;
}

__device__ static inline bool IsValidCapability(const unsigned res[8]) {
    siphash_keys keys = {
        .k0 = ((unsigned long)res[1] << 32) | res[0],
        .k1 = ((unsigned long)res[3] << 32) | res[2],
        .k2 = (0x0adb042aull << 32) | res[4],
        .k3 = 0x5b13db8630a80964ull,
    };
    diphash_state ds(keys);
    ds.hash24();
    unsigned long carried_mac = ((unsigned long)res[7] << 32) | res[6];
    unsigned long mac = ds.xor_lanes();
    return carried_mac == mac;
}
} // namespace rpc
} // namespace gpumpc