#include "rfc5514.h"
#include "rfc5514_impl.h"

#include "runtime-gpu/core/process_control_block.h"
#include "utils/filesystem.h"
#include "utils/hip_helper.h"
#include "utils/monad_runner.h"

#include <absl/status/status.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <random>

DEFINE_string(runtime_bin, "", "Location of the runtime binary");

using namespace gpumpc;
using namespace gpumpc::runtime;

class DiffieHellmanTest : public ::testing::Test {
  protected:
    virtual void SetUp() override;
    virtual void TearDown() override;
    hipModule_t module_;
    hipFunction_t f_dh_exchange_;
    void *scratch_;
};

void DiffieHellmanTest::SetUp() {
    absl::Status stat;
    auto data = gpumpc::ReadAll(FLAGS_runtime_bin, &stat);
    ASSERT_TRUE(stat.ok());
    MonadRunner<hipError_t> runner(hipSuccess);
    auto s =
        runner.Run([&]() { return hipModuleLoadData(&module_, data.data()); })
            .Run([&]() {
                return hipModuleGetFunction(&f_dh_exchange_, module_,
                                            "DHKeyExchange");
            })
            .Run([&]() {
                return hipMalloc(
                    &scratch_,
                    sizeof(ProcessControlBlock) +
                        sizeof(BigInt<ProcessControlBlock::
                                          kDiffieHellmanSharedSecretBits>));
            });
    ASSERT_EQ(s.code(), hipSuccess);
}

void DiffieHellmanTest::TearDown() {
    auto s = hipFree(scratch_);
    ASSERT_EQ(s, hipSuccess);
}

TEST_F(DiffieHellmanTest, TestDH) {
    using Mont = MontNum<RFC5514PrimeField>;
    auto g = RFC5514PrimeField::G();
    Mont mg = Mont::Reduce(g);

#pragma pack(push, 1)
    struct {
        ProcessControlBlock pcb;
        Mont mgb;
    } scratch;
#pragma pack(pop)
    memset(&scratch, 0, sizeof(scratch));
    Array<unsigned, scratch.pcb.dh_seed.size()> b;
    std::default_random_engine gen;
    gen.seed(1);
    std::uniform_int_distribution<> distrib(
        0, std::numeric_limits<unsigned>::max());
    for (int i = 0; i < 8; i++) {
        scratch.pcb.dh_seed[i] = distrib(gen);
        b[i] = distrib(gen);
    }
    BigInt<256> bn_a{scratch.pcb.dh_seed}, bn_b{b};
    bn_a.Dump();
    bn_b.Dump();
    scratch.mgb = mg.Pow(bn_b);

    auto shared_secret = mg.Pow(bn_a).Pow(bn_b);
    shared_secret.Dump();

    auto stat = hipMemcpyHtoD(scratch_, &scratch, sizeof(scratch));
    ASSERT_EQ(stat, hipSuccess);
    struct {
        void *pcb;
        void *mgb;
    } param = {
        .pcb = scratch_,
        .mgb = reinterpret_cast<char *>(scratch_) + sizeof(ProcessControlBlock),
    };
    auto s = experiment::LaunchKernel(f_dh_exchange_, 1, 1, param);
    ASSERT_TRUE(s.ok());
    stat = hipMemcpyDtoH(&scratch, scratch_, sizeof(scratch));
    ASSERT_EQ(stat, hipSuccess);
    BigInt<2048> dh_shared_secret{scratch.pcb.dh_shared_secret};
    dh_shared_secret.Dump();
    ASSERT_EQ(memcmp(shared_secret.Data().Digits().data(),
                     scratch.pcb.dh_shared_secret.data(),
                     sizeof(shared_secret)),
              0);
}

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}