#pragma once

#include "runtime-gpu/adt/array.h"
#include "runtime-gpu/core/abi.h"

#include <array>
#include <iostream>

namespace gpumpc {

template <unsigned kBits> struct BigInt {
    enum { kWords = kBits / 32 };
    GPUMPC_HOST_AND_DEVICE static BigInt One();
    GPUMPC_HOST_AND_DEVICE static BigInt From(const unsigned *data);
    GPUMPC_HOST_AND_DEVICE static void MulLo(BigInt *res, const BigInt &lhs,
                                             const BigInt &rhs);
    GPUMPC_HOST_AND_DEVICE static void MulHi(BigInt *res, const BigInt &lhs,
                                             const BigInt &rhs);
    GPUMPC_HOST_AND_DEVICE static void AddWithCC(BigInt *res, const BigInt &lhs,
                                                 const BigInt &rhs,
                                                 unsigned initial_carry);
    GPUMPC_HOST_AND_DEVICE bool operator>=(const BigInt &rhs) const;
    void Dump() const;
    GPUMPC_HOST_AND_DEVICE const Array<unsigned, kWords> &Digits() const {
        return data_;
    }

    Array<unsigned, kWords> data_;
};

template <unsigned kBits>
GPUMPC_HOST_AND_DEVICE bool
BigInt<kBits>::operator>=(const BigInt<kBits> &rhs) const {
    for (int i = BigInt<kBits>::kWords; i--;) {
        if (data_[i] > rhs.data_[i]) {
            return true;
        } else if (data_[i] < rhs.data_[i]) {
            return false;
        }
    }
    return false;
}

template <unsigned kBits>
inline BigInt<kBits> GPUMPC_HOST_AND_DEVICE BigInt<kBits>::One() {
    BigInt bn;
    for (unsigned i = 0; i < kWords; i++) {
        bn.data_[i] = 0;
    }
    bn.data_[0] = 1;
    return bn;
}

template <unsigned kBits>
GPUMPC_HOST_AND_DEVICE inline BigInt<kBits>
BigInt<kBits>::From(const unsigned *data) {
    BigInt bn;
    for (unsigned i = 0; i < kWords; i++) {
        bn.data_[i] = data[i];
    }
    return bn;
}

template <unsigned kBits>
GPUMPC_HOST_AND_DEVICE inline void
BigInt<kBits>::MulLo(BigInt *res, const BigInt &lhs, const BigInt &rhs) {
    unsigned r[kWords] = {
        0,
    };

#ifdef __HIP_DEVICE_COMPILE__
#pragma unroll 1
#endif
    for (int i = 0; i < kWords; i++) {
        unsigned carry = 0;
#ifdef __HIP_DEVICE_COMPILE__
#pragma unroll 1
#endif
        for (int j = 0; i + j < kWords; j++) {
            unsigned long v =
                (unsigned long)lhs.data_[j] * rhs.data_[i] + carry + r[i + j];
            r[i + j] = (unsigned)v;
            carry = v >> 32;
        }
    }
    for (int i = 0; i < kWords; i++) {
        res->data_[i] = r[i];
    }
}

template <unsigned kBits>
GPUMPC_HOST_AND_DEVICE inline void
BigInt<kBits>::MulHi(BigInt *res, const BigInt &lhs, const BigInt &rhs) {
    unsigned r[kWords] = {
        0,
    };
#ifdef __HIP_DEVICE_COMPILE__
#pragma unroll 1
#endif
    for (int i = 0; i < kWords; i++) {
        unsigned carry = 0;
#ifdef __HIP_DEVICE_COMPILE__
#pragma unroll 1
#endif
        for (int j = 0; j < kWords; j++) {
            unsigned idx = (i + j) % kWords;
            unsigned long v =
                (unsigned long)lhs.data_[j] * rhs.data_[i] + carry + r[idx];
            r[idx] = (unsigned)v;
            carry = v >> 32;
        }
        r[i] = carry;
    }
    for (int i = 0; i < kWords; i++) {
        res->data_[i] = r[i];
    }
}

template <unsigned kBits>
GPUMPC_HOST_AND_DEVICE inline void
BigInt<kBits>::AddWithCC(BigInt *res, const BigInt &lhs, const BigInt &rhs,
                         unsigned initial_carry) {
    unsigned carry = initial_carry;
    for (int i = 0; i < kWords; i++) {
        unsigned long v = (unsigned long)lhs.data_[i] + rhs.data_[i] + carry;
        res->data_[i] = (unsigned)v;
        carry = v >> 32;
    }
}

template <unsigned kBits> void BigInt<kBits>::Dump() const {
    for (int i = kWords; i--;) {
        printf("%08x ", data_[i]);
    }
    printf("\n");
}

} // namespace gpumpc