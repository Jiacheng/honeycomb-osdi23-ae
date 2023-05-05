#include "conv2d.h"
#include "winograd.h"
#include <gtest/gtest.h>
#include <vector>
#include <random>

typedef Conv2dModule<float, 128, 128, 128, 2, 2, 3, 3, 1, 1, 1, 1> Conv2dType;

class WinogradTest : public ::testing::Test {
  protected:
	virtual void SetUp() override;
	virtual void TearDown() override {}

	std::vector<float> input;
	std::vector<float> kernel;
	std::vector<float> output;
	std::vector<float> output_comp;
};

void WinogradTest::SetUp() {
	input.resize(
			Conv2dType::BATCH * Conv2dType::IN_CHANNEL * Conv2dType::IN_H
					* Conv2dType::IN_W);
	kernel.resize(
			Conv2dType::OUT_CHANNEL * Conv2dType::IN_CHANNEL
					* Conv2dType::KERNEL_H * Conv2dType::KERNEL_W);
	output.resize(
			Conv2dType::BATCH * Conv2dType::OUT_CHANNEL * Conv2dType::OUT_H
					* Conv2dType::OUT_W);
	output_comp.resize(output.size());

	std::srand(0);

	for (int nn = 0; nn < Conv2dType::BATCH; ++nn) {
		for (int rc = 0; rc < Conv2dType::IN_CHANNEL; ++rc) {
			for (int j = 0; j < Conv2dType::IN_H; ++j) {
				for (int i = 0; i < Conv2dType::IN_W; ++i) {
					const auto idx = ((((nn * Conv2dType::IN_CHANNEL) + rc)
							* Conv2dType::IN_H) + j) * Conv2dType::IN_W + i;
					input[idx] = rand() / (float) RAND_MAX;
				}
			}
		}
	}

	for (int ff = 0; ff < Conv2dType::OUT_CHANNEL; ++ff) {
		for (int rc = 0; rc < Conv2dType::IN_CHANNEL; ++rc) {
			for (int ry = 0; ry < Conv2dType::KERNEL_H; ++ry) {
				for (int rx = 0; rx < Conv2dType::KERNEL_W; ++rx) {
					const auto idx = ((((ff * Conv2dType::IN_CHANNEL) + rc)
							* Conv2dType::KERNEL_H) + ry) * Conv2dType::KERNEL_W
							+ rx;
					kernel[idx] = rand() / (float) RAND_MAX;
				}
			}
		}
	}
}

TEST_F(WinogradTest, TestWinogradInference) {
	Conv2dType conv2d;
	conv2d.Apply(input.data(), kernel.data(), output_comp.data());

	winograd_conv2d_nchw_128_128_2_3_1_init(output.data());
	winograd_conv2d_nchw_128_128_2_3_1(input.data(), kernel.data(), output.data());

	const float error = 1e-3;
	for (std::size_t i = 0; i < output.size(); ++i) {
		EXPECT_NEAR(output_comp[i], output[i], error);
	}
}
