#ifndef EXPERIMENTS_RESNET_CONV2D_H_
#define EXPERIMENTS_RESNET_CONV2D_H_

template<class T, int _BATCH, int _IN_CHANNEL, int _OUT_CHANNEL, int _IN_H, int _IN_W, int _KERNEL_H, int _KERNEL_W, int STRIDE_H, int STRIDE_W,
		int _PADDING_H, int _PADDING_W, int DILATION_H = 1, int DILATION_W = 1>
class Conv2dModule {
public:
	static constexpr int BATCH = _BATCH;
	static constexpr int IN_CHANNEL = _IN_CHANNEL;
	static constexpr int OUT_CHANNEL = _OUT_CHANNEL;
	static constexpr int IN_H = _IN_H;
	static constexpr int IN_W = _IN_W;
	static constexpr int KERNEL_H = _KERNEL_H;
	static constexpr int KERNEL_W = _KERNEL_W;
	static constexpr int PADDING_H = _PADDING_H;
	static constexpr int PADDING_W = _PADDING_W;
	static constexpr int OUT_H = (IN_H - ((KERNEL_H - 1) * DILATION_H + 1) + PADDING_H * 2) / STRIDE_H + 1;
	static constexpr int OUT_W = (IN_W - ((KERNEL_W - 1) * DILATION_W + 1) + PADDING_W * 2) / STRIDE_W + 1;

	static void Apply(const T *in, const T *kernel, T *out) {
		for (int i = 0; i < BATCH * OUT_CHANNEL * OUT_H * OUT_W; ++i) {
			out[i] = 0;
		}

		for (int rc = 0; rc < IN_CHANNEL; ++rc) {
			for (int ry = 0; ry < KERNEL_H; ++ry) {
				for (int rx = 0; rx < KERNEL_W; ++rx) {
					for (int ff = 0; ff < OUT_CHANNEL; ++ff) {
						for (int nn = 0; nn < BATCH; ++nn) {
							for (int yy = 0; yy < OUT_H; ++yy) {
								for (int xx = 0; xx < OUT_W; ++xx) {
									const int jj = yy * STRIDE_H + ry * DILATION_H;
									const int j = jj - PADDING_H;
									if (j >= 0 && j < IN_H) {
										const int ii = xx * STRIDE_W
												+ rx * DILATION_W;
										const int i = ii - PADDING_W;
										if (i >= 0 && i < IN_W) {
											out[((((nn * OUT_CHANNEL) + ff) * OUT_H) + yy) * OUT_W + xx] +=
													in[((((nn * IN_CHANNEL) + rc) * IN_H) + j) * IN_W + i] *
													kernel[((((ff * IN_CHANNEL) + rc) * KERNEL_H) + ry) * KERNEL_W + rx];
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
};

#endif /* EXPERIMENTS_RESNET_CONV2D_H_ */
