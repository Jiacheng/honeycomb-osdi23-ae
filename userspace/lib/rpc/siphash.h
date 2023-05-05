#pragma once

#include <hip/hip_runtime.h>

typedef struct {
    unsigned long k0;
    unsigned long k1;
    unsigned long k2;
    unsigned long k3;
} siphash_keys;

__device__ __forceinline__ uint64_t devectorize(uint2 x) {
    return ((unsigned long)x.y << 32) | x.x;
}

__device__ __forceinline__ uint2 vectorize(const uint64_t x) {
    uint2 result;
    result.x = (unsigned)x;
    result.y = x >> 32;
    return result;
}

class diphash_state {
  public:
    using sip64 = uint2;
    sip64 v0;
    sip64 v1;
    sip64 v2;
    sip64 v3;

    __device__ diphash_state(const siphash_keys &sk) {
        v0 = vectorize(sk.k0);
        v1 = vectorize(sk.k1);
        v2 = vectorize(sk.k2);
        v3 = vectorize(sk.k3);
    }

    __device__ diphash_state(const sip64 sk[4]) {
        v0 = sk[0];
        v1 = sk[1];
        v2 = sk[2];
        v3 = sk[3];
    }

    __device__ uint64_t xor_lanes() {
        return devectorize((v0 ^ v1) ^ (v2 ^ v3));
    }

    __device__ void xor_with(const diphash_state &x) {
        v0 ^= x.v0;
        v1 ^= x.v1;
        v2 ^= x.v2;
        v3 ^= x.v3;
    }

    __device__ void dip_round() {
        v0 += v1;
        v2 += v3;
        v1 = rotl(v1, 13);
        v3 = rotl(v3, 16);
        v1 ^= v0;
        v3 ^= v2;
        v0 = rotl(v0, 32);
        v2 += v1;
        v0 += v3;
        v1 = rotl(v1, 17);
        v3 = rotl(v3, 21);
        v1 ^= v2;
        v3 ^= v0;
        v2 = rotl(v2, 32);
    }

    __device__ void dip_round0() {
        v2 += v3;
        v3 = rotl(v3, 16);
        v3 ^= v2;
        v2 += v1;
        v0 += v3;
        v1 = rotl(v1, 17);
        v3 = rotl(v3, 21);
        v1 ^= v2;
        v3 ^= v0;
        v2 = rotl(v2, 32);
    }

    __device__ void hash24() {
        dip_round0();
        dip_round();
        v2 ^= vectorize(0xff);
        for (int i = 0; i < 4; i++) {
            dip_round();
        }
    }

    __device__ static uint2 rotl(uint2 r, unsigned n) {
        unsigned long v = devectorize(r);
        unsigned long ret = (v << n) | (v >> (64 - n));
        return vectorize(ret);
    }
};