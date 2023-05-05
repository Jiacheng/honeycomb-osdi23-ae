#include "montgomery.h"
#include "rfc5514.h"
#include "runtime-gpu/core/process_control_block.h"

#include "rfc5514_impl.h"

namespace gpumpc {
namespace runtime {

extern "C" {
__global__ void DHKeyExchange(ProcessControlBlock *pcb,
                              const MontNum<RFC5514PrimeField> *mgb);
}

/**
 * A variant of Diffie-Hellman key exchange to establish a shared secret between
 * the CPU and the GPU. It uses Montgomery reduction to speed up the
 * computation.
 *
 * Inputs:
 *   * The PCB contains the private value a.
 *   * gb: The value of g^b in Montgomery domain.
 *
 * Output:
 *   * The shared secret in Montgomery domain, written back into the PCB.
 */
__global__ void DHKeyExchange(ProcessControlBlock *pcb,
                              const MontNum<RFC5514PrimeField> *mgb) {
    using Mont = MontNum<RFC5514PrimeField>;
    using BN = BigInt<ProcessControlBlock::kDiffieHellmanSharedSecretBits>;
    BigInt<ProcessControlBlock::kDiffieHellmanSeedBits> a{pcb->dh_seed};
    Mont secret = mgb->Pow(a);
    pcb->dh_shared_secret = secret.Data().Digits();
}

} // namespace runtime
} // namespace gpumpc