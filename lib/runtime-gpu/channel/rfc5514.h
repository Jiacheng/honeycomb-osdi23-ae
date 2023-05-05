#pragma once

#include "bigint.h"
#include "montgomery.h"
#include "runtime-gpu/adt/array.h"
#include "runtime-gpu/core/abi.h"

namespace gpumpc {
class RFC5514PrimeField {
  public:
    using Number = BigInt<2048>;
    static constexpr unsigned RLog2() { return 2048; }
    GPUMPC_HOST_AND_DEVICE static Number R() { return r_; }
    GPUMPC_HOST_AND_DEVICE static Number R2() { return r2_; }
    GPUMPC_HOST_AND_DEVICE static const Number &RPrime() { return r_prime_; }
    GPUMPC_HOST_AND_DEVICE static const Number &G() { return g_; }
    GPUMPC_HOST_AND_DEVICE static const Number &N() { return n_; }
    GPUMPC_HOST_AND_DEVICE static const Number &MinusN() { return minus_n_; }
    GPUMPC_HOST_AND_DEVICE static const Number &NPrime() { return n_prime_; }
    GPUMPC_HOST_AND_DEVICE static const MontNum<RFC5514PrimeField> &
    MontRawR2() {
        return raw_r2_;
    }

    GPUMPC_CONSTANT static const Number r_;
    GPUMPC_CONSTANT static const Number r2_;
    GPUMPC_CONSTANT static const Number r_prime_;
    GPUMPC_CONSTANT static const Number n_; //;
    GPUMPC_CONSTANT static const Number minus_n_;
    GPUMPC_CONSTANT static const Number n_prime_;
    GPUMPC_CONSTANT static const MontNum<RFC5514PrimeField> raw_r2_;
    GPUMPC_CONSTANT static const Number g_;
};

} // namespace gpumpc