#include "crypto/sha3.h"
#include "ethash_solver.h"
#include "experiments/platform.h"

#include <functional>
#include <gtest/gtest.h>
#include <initializer_list>
#include <string_view>

using namespace gpumpc::experiment;
using namespace gpumpc::experiment::ethash;

class EthashTest : public ::testing::Test {
  protected:
    virtual void SetUp() override;
    virtual void TearDown() override;
};

void EthashTest::SetUp() {
    auto stat = ExperimentPlatform::Instance().Initialize();
    ASSERT_TRUE(stat.ok());
}

void EthashTest::TearDown() {
    auto stat = ExperimentPlatform::Instance().Close();
    ASSERT_TRUE(stat.ok());
}

TEST_F(EthashTest, TestSolveEthash) {
    const uint64_t h_header[4] = {0xb16e8fbf4287a6b8ull, 0xd73bb92fd8697fe2ull,
                                  0xaba5ab4537296914ull, 0x65c169cbd1206137ull};
    uint64_t target = 0xfffffffffffUL;

    EthashSolver solver;
    auto stat = solver.Initialize();
    ASSERT_TRUE(stat.ok());
    solver.InitDAG(200);

    std::array<unsigned, kEthashDualSolverSearchResults> results;
    absl::Span<unsigned> sols(results);
    solver.SetJob(h_header);
    stat = solver.Solve(0, target);
    ASSERT_TRUE(stat.ok());

    size_t dummy = 0;
    auto ret = solver.Fetch(&sols, &dummy);
    ASSERT_EQ(1, ret);
    ASSERT_EQ(0x23e6f, sols[0]);
}