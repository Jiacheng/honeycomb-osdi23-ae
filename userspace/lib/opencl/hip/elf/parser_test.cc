#include <gtest/gtest.h>
#include "amdgpu_program.h"
#include "utils/filesystem.h"

static const char *sKernelFile;

TEST(TestParseAMDGPUProgram, TestParse) {
    using namespace ocl::hip;
    absl::Status stat;
    std::vector<char> data = gpumpc::ReadAll(sKernelFile, &stat);
    ASSERT_TRUE(stat.ok());

    auto prog = ParseAMDGPUProgram(std::string_view(data.data(), data.size()), &stat);
    ASSERT_TRUE(stat.ok());
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <kernel>\n";
        return -1;
    }
    sKernelFile = argv[1];
    return RUN_ALL_TESTS();
}
