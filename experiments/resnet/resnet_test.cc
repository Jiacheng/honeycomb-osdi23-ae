#include "crypto/sha3.h"
#include "experiments/platform.h"
#include "experiments/resnet/resnet_inference.h"
#include <absl/types/span.h>
#include <gtest/gtest.h>
#include <initializer_list>

using namespace gpumpc::experiment;
using namespace gpumpc;

class ResNetTest : public ::testing::Test {
  protected:
    virtual void SetUp() override;
    virtual void TearDown() override;
};

void ResNetTest::SetUp() {
    auto stat = ExperimentPlatform::Instance().Initialize();
    ASSERT_TRUE(stat.ok());
}

void ResNetTest::TearDown() {
    auto stat = ExperimentPlatform::Instance().Close();
    ASSERT_TRUE(stat.ok());
}

TEST_F(ResNetTest, TestResNet1Inference) {
    static const char kImageName[] = "data/resnet/image/image.dat";
    static const uint64_t kExpectedHash[] = {
        0x04cafeb01eac2371ull,
        0x6a885323218c225aull,
        0xab25ab59548256b7ull,
        0xa0d936ee0f7468b1ull,
    };

    std::vector<char> image;
    auto &plat = ExperimentPlatform::Instance();
    auto stat = plat.LoadResource(kImageName, &image);
    ASSERT_TRUE(stat.ok());
    std::vector<char> original_result(ResNetInference::kResultSize);
    {
        auto original_model = NewResNet1Baseline();
        stat = original_model->Initialize();
        ASSERT_TRUE(stat.ok());
        stat = original_model->Run(image);
        ASSERT_TRUE(stat.ok());
        stat = original_model->Fetch(
            absl::MakeSpan(original_result.data(), original_result.size()));
        ASSERT_TRUE(stat.ok());
        stat = original_model->Close();
        ASSERT_TRUE(stat.ok());

        uint64_t sha3[4];
        SHA3_256(reinterpret_cast<unsigned char *>(sha3),
                 reinterpret_cast<unsigned char *>(original_result.data()),
                 original_result.size());
        ASSERT_EQ(absl::MakeConstSpan(kExpectedHash),
                  absl::MakeConstSpan(sha3));
    }

    {
        std::vector<char> result(ResNetInference::kResultSize);
        auto model = NewResNet1();
        stat = model->Initialize();
        ASSERT_TRUE(stat.ok());
        stat = model->Run(image);
        ASSERT_TRUE(stat.ok());
        stat = model->Fetch(absl::MakeSpan(result.data(), result.size()));
        ASSERT_TRUE(stat.ok());
        stat = model->Close();
        ASSERT_TRUE(stat.ok());

        const float error = 1e-3;
        for (unsigned i = 0; i < result.size(); i += sizeof(float)) {
            EXPECT_NEAR(*(const float *)&original_result[i],
                        *(const float *)&result[i], error);
        }
    }
}

TEST_F(ResNetTest, TestResNet18Inference) {
    enum { kResultSize = 4000 };
    auto model = NewResNet18();
    static const char kImageName[] = "data/resnet/image/image_1.dat";
    static const char kRefName[] = "data/resnet/output/resnet18_image_1.dat";
    auto &plat = ExperimentPlatform::Instance();
    std::vector<char> reference;
    auto stat = plat.LoadResource(kRefName, &reference);
    ASSERT_TRUE(stat.ok() && reference.size() == kResultSize);
    std::vector<char> image;
    stat = plat.LoadResource(kImageName, &image);
    ASSERT_TRUE(stat.ok());

    stat = model->Initialize();
    ASSERT_TRUE(stat.ok());
    stat = model->Run(image);
    ASSERT_TRUE(stat.ok());
    std::vector<float> res(kResultSize / sizeof(float));
    stat = model->Fetch(absl::MakeSpan(reinterpret_cast<char *>(res.data()),
                                       res.size() * sizeof(float)));
    ASSERT_TRUE(stat.ok());
    stat = model->Close();
    ASSERT_TRUE(stat.ok());

    auto ref = reinterpret_cast<const float *>(reference.data());
    for (size_t i = 0; i < res.size(); i++) {
        ASSERT_NEAR(res[i], ref[i], 1e-5);
    }
}
