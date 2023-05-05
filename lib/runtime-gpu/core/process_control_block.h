#pragma once

#include "runtime-gpu/adt/array.h"

namespace gpumpc {
namespace runtime {

/**
 * A per-process control block that reside in the hidden VMA region.
 * It stores the state required to set up secure communication channels, etc.
 */
struct ProcessControlBlock {
  public:
    enum {
        kDiffieHellmanSeedBits = 256,
        kDiffieHellmanSharedSecretBits = 2048,
    };

    Array<unsigned, kDiffieHellmanSeedBits / 32> dh_seed;
    Array<unsigned, kDiffieHellmanSharedSecretBits / 32> dh_shared_secret;
};
} // namespace runtime
} // namespace gpumpc