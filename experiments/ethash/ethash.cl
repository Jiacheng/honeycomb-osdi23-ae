#include "keccak.clh"
#include "dualsolver_types.h"

#define FNV_PRIME 0x01000193U
#define fnv(x, y) ((x)*FNV_PRIME ^ (y))
#define fnv_reduce(v) fnv(fnv(fnv(v.x, v.y), v.z), v.w)

typedef union {
    uint uints[128 / sizeof(uint)];
    ulong ulongs[128 / sizeof(ulong)];
    uint2 uint2s[128 / sizeof(uint2)];
    uint4 uint4s[128 / sizeof(uint4)];
    uint8 uint8s[128 / sizeof(uint8)];
    uint16 uint16s[128 / sizeof(uint16)];
    ulong8 ulong8s[128 / sizeof(ulong8)];
} hash128_t;

typedef union {
    ulong8 ulong8s[1];
    ulong4 ulong4s[2];
    uint2 uint2s[8];
    uint4 uint4s[4];
    uint8 uint8s[2];
    uint16 uint16s[1];
    ulong ulongs[8];
    uint uints[16];
} compute_hash_share;

static void keccak_512(uint2 *s) {
    uint2 st[25];

    for (uint i = 0; i < 8; ++i)
        st[i] = s[i];

    st[8] = (uint2)(0x00000001, 0x80000000);

    for (uint i = 9; i != 25; ++i)
        st[i] = (uint2)(0);

    KECCAK_PROCESS(st, 8);

    for (uint i = 0; i < 8; ++i)
        s[i] = st[i];
}

__kernel void InitializeEthashDAG(__global uint16 *restrict _g_dag,
                                  __global const uint16 *restrict _g_light,
                                  uint start, uint dag_size, uint light_size) {
    enum {
        NODE_WORDS = 16,
        ETHASH_DATASET_PARENTS = 256,
    };

    typedef union {
        uint words[64 / sizeof(uint)];
        uint2 uint2s[64 / sizeof(uint2)];
        uint4 uint4s[64 / sizeof(uint4)];
    } hash64_t;

    __global const hash64_t *g_light = (__global const hash64_t *)_g_light;
    __global hash64_t *g_dag = (__global hash64_t *)_g_dag;
    uint const node_index = start + get_global_id(0);

    if (node_index >= dag_size * 2) {
        return;
    }

    hash64_t dag_node = g_light[node_index % light_size];
    dag_node.words[0] ^= node_index;
    keccak_512(dag_node.uint2s);

#pragma unroll NODE_WORDS
    for (uint i = 0; i < ETHASH_DATASET_PARENTS; ++i) {
        uint parent_index =
            fnv(node_index ^ i, dag_node.words[i % NODE_WORDS]) % light_size;
        __global const hash64_t *parent = g_light + parent_index;

#pragma unroll
        for (uint x = 0; x < 4; ++x) {
            dag_node.uint4s[x] *= (uint4)(FNV_PRIME);
            dag_node.uint4s[x] ^= parent->uint4s[x];
        }
    }

    keccak_512(dag_node.uint2s);
    g_dag[node_index] = dag_node;
}

static uint fast_mod(uint dividend, uint divisor, uint precompute_mult,
                     uint precompute_shift) {
    uint2 r = as_uint2((ulong)dividend * precompute_mult);
    uint t = ((dividend - r.y) >> 1) + r.y;
    uint q = t >> precompute_shift;
    uint m = q * divisor;
    uint res = dividend - m;
    return res;
}

// All accesses to the shared memory are synchronized per-wave.
// In GCN all threads in the waves are executed in locksteps.
// Therefore a mem_fence is sufficient -- or it can be completely eliminiated if
// kEthashOnlyBlockSize == 64 (the compiler should remove it).
// We still need to define it to avoid reordering

#define SYNCWAVE() mem_fence(CLK_LOCAL_MEM_FENCE)

__attribute__((reqd_work_group_size(kEthashBlockSize, 1, 1)))
__kernel void EthashSingleSearch(const uint8 header,
                                 __global const uint16 *restrict _g_dag,
                                 __global uint4 *restrict output,
                                 ulong start_nonce, ulong target, uint dag_size,
                                 uint dag_size_shift,
                                 uint dag_size_multiplier) {
    enum {
        ACCESSES = 64,
        THREADS_PER_HASH = 128 / 16,
        PARALLEL_HASH = kEthashHashParallelism,
        kBlockSize = kEthashBlockSize,
    };

    __global struct EthashSolutionsOnDevice *g_output =
        (__global struct EthashSolutionsOnDevice *)output;
    __global hash128_t const *g_dag = (__global hash128_t const *)_g_dag;

    const uint lid = get_local_id(0);
    const uint thread_id = lid % THREADS_PER_HASH;
    const uint hash_id = lid / THREADS_PER_HASH;
    const int mix_idx = lid & 3;
    const uint gid = get_global_id(0);

    __local compute_hash_share
        sharebuf[kBlockSize / THREADS_PER_HASH];
    __local uint buffer[kBlockSize];
    __local compute_hash_share *const share = sharebuf + hash_id;

    // sha3_512(header .. nonce)
    uint2 state[25];
    state[0] = header.s01;
    state[1] = header.s23;
    state[2] = header.s45;
    state[3] = header.s67;
    state[4] = as_uint2(start_nonce + gid);
    state[5] = as_uint2(0x0000000000000001UL);
    state[6] = (uint2)(0);
    state[7] = (uint2)(0);
    state[8] = as_uint2(0x8000000000000000UL);
#pragma unroll
    for (int i = 9; i < 25; ++i) {
        state[i] = (uint2)(0);
    }

    uint init0;

    keccak_process(state, 8);

    uint4 mix;

#pragma unroll 1
    for (uint tid = 0; tid < THREADS_PER_HASH; tid++) {
        if (tid == thread_id) {
            for (int i = 0; i < 8; ++i) {
                share->uint2s[i] = state[i];
            }
        }

        SYNCWAVE();

        mix = share->uint4s[mix_idx];
        init0 = share->uints[0];

        SYNCWAVE();

#pragma unroll
        for (uint a = 0; a < ACCESSES; a += 4) {
            const uint lane_idx =
                THREADS_PER_HASH * hash_id + a / 4 % THREADS_PER_HASH;
#pragma unroll
            for (uint x = 0; x < 4; ++x) {
                uint s = ((uint *)&mix)[x];
                uint v = fnv(init0 ^ (a + x), s);
                uint res =
                    fast_mod(v, dag_size, dag_size_multiplier, dag_size_shift);

                buffer[lid] = res;
                SYNCWAVE();
                mix = fnv(mix, g_dag[buffer[lane_idx]].uint4s[thread_id]);
                SYNCWAVE();
            }
        }

        share->uints[thread_id] = fnv_reduce(mix);

        SYNCWAVE();

        if (tid == thread_id) {
            for (int i = 0; i < 4; ++i) {
                state[i + 8] = share->uint2s[i];
            }
        }
    }

    state[12] = as_uint2(0x0000000000000001UL);
    state[13] = (uint2)(0);
    state[14] = (uint2)(0);
    state[15] = (uint2)(0);
    state[16] = as_uint2(0x8000000000000000UL);
#pragma unroll
    for (int i = 17; i < 25; ++i) {
        state[i] = (uint2)(0);
    }
    keccak_process(state, 1);

    if (as_ulong(as_uchar8(state[0]).s76543210) <= target) {
        uint slot = min(kEthashDualSolverSearchResults - 1u,
                        atomic_inc(&g_output->count));
        g_output->gid[slot] = gid;
    }
}
