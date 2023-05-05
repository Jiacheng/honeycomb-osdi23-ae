#define GET_GROUP_ID() ( \
	(get_group_id(0) * get_num_groups(1) + get_group_id(1)) * get_num_groups(2) + get_group_id(2) \
)

#define GET_NUM_GROUPS() (get_num_groups(0) * get_num_groups(1) * get_num_groups(2))

#define GET_LOCAL_ID() ( \
	(get_local_id(0) * get_local_size(1) + get_local_id(1)) * get_local_size(2) + get_local_id(2) \
)

#define GET_LOCAL_SIZE() (get_local_size(0) * get_local_size(1) * get_local_size(2))

#define GET_GLOBAL_ID() ( \
	(get_global_id(0) * get_global_size(1) + get_global_id(1)) * get_global_size(2) + get_global_id(2) \
)

#define GET_GLOBAL_SIZE() (get_global_size(0) * get_global_size(1) * get_global_size(2))

#define DEFINE_CONV2D_ARGS_WITH_DILATIONS(_BATCH, _IN_CHANNEL, \
	_OUT_CHANNEL, _IN_H, _IN_W, _KERNEL_H, _KERNEL_W, _STRIDE_H, \
	_STRIDE_W,	_PADDING_H, _PADDING_W, _DILATION_H, _DILATION_W) \
		const int BATCH = (_BATCH); \
		const int IN_CHANNEL = (_IN_CHANNEL); \
		const int OUT_CHANNEL = (_OUT_CHANNEL); \
		const int IN_H = (_IN_H); \
		const int IN_W = (_IN_W); \
		const int KERNEL_H = (_KERNEL_H); \
		const int KERNEL_W = (_KERNEL_W); \
		const int STRIDE_H = (_STRIDE_H); \
		const int STRIDE_W = (_STRIDE_W); \
		const int PADDING_H = (_PADDING_H); \
		const int PADDING_W = (_PADDING_W); \
		const int DILATION_W = (_DILATION_W); \
		const int DILATION_H = (_DILATION_H); \
		const int OUT_H = (IN_H - ((KERNEL_H - 1) * DILATION_H + 1) + PADDING_H * 2) / STRIDE_H + 1; \
		const int OUT_W = (IN_W - ((KERNEL_W - 1) * DILATION_W + 1) + PADDING_W * 2) / STRIDE_W + 1;

#define DEFINE_CONV2D_ARGS(_BATCH, _IN_CHANNEL, \
	_OUT_CHANNEL, _IN_H, _IN_W, _KERNEL_H, _KERNEL_W, _STRIDE_H, \
	_STRIDE_W,	_PADDING_H, _PADDING_W) \
		DEFINE_CONV2D_ARGS_WITH_DILATIONS((_BATCH), (_IN_CHANNEL), \
			(_OUT_CHANNEL), (_IN_H), (_IN_W), (_KERNEL_H), (_KERNEL_W), \
			(_STRIDE_H), (_STRIDE_W), (_PADDING_H), (_PADDING_W), 1, 1)

#define CONV2D_LOOPS do { \
	for (int nn = get_global_id(0); nn < BATCH; nn += get_global_size(0)) { \
		for (int ff = get_global_id(1); ff < OUT_CHANNEL; ff += get_global_size(1)) { \
			for (int yy = get_global_id(2); yy < OUT_H; yy += get_global_size(2)) { \
				for (int xx = 0; xx < OUT_W; ++xx) { \
					out[((((nn * OUT_CHANNEL) + ff) * OUT_H) + yy) * OUT_W + xx] = 0; \
				} \
			} \
		} \
	} \
\
	for (int rc = 0; rc < IN_CHANNEL; ++rc) { \
		for (int ry = 0; ry < KERNEL_H; ++ry) { \
			for (int rx = 0; rx < KERNEL_W; ++rx) { \
				for (int nn = get_global_id(0); nn < BATCH; nn += get_global_size(0)) { \
					for (int ff = get_global_id(1); ff < OUT_CHANNEL; ff += get_global_size(1)) { \
						for (int yy = get_global_id(2); yy < OUT_H; yy += get_global_size(2)) { \
							for (int xx = 0; xx < OUT_W; ++xx) { \
								const int jj = yy * STRIDE_H + ry * DILATION_H; \
								const int j = jj - PADDING_H; \
								if (j >= 0 && j < IN_H) { \
									const int ii = xx * STRIDE_W + rx * DILATION_W; \
									const int i = ii - PADDING_W; \
									if (i >= 0 && i < IN_W) { \
										out[((((nn * OUT_CHANNEL) + ff) * OUT_H) + yy) * OUT_W + xx] += \
												in[((((nn * IN_CHANNEL) + rc) * IN_H) + j) * IN_W + i] * \
												ker[((((ff * IN_CHANNEL) + rc) * KERNEL_H) + ry) * KERNEL_W + rx]; \
									} \
								} \
							} \
						} \
					} \
				} \
			} \
		} \
	} \
} while(0)

__kernel void convolution_forward_replacement1(
		__global const float *ker,
		__global const float *in,
		__global float *out,
		__global const void *p_desc_tuple) {
	DEFINE_CONV2D_ARGS(128, 64, 128, 3, 3, 3, 3, 2, 2, 1, 1);
	CONV2D_LOOPS;
}

__kernel void convolution_forward_replacement2(
		__global const float *ker,
		__global const float *in,
		__global float *out,
		__global const void *p_desc_tuple) {
	DEFINE_CONV2D_ARGS(128, 64, 128, 3, 3, 1, 1, 2, 2, 0, 0);
	CONV2D_LOOPS;
}

__kernel void winograd_conv2d_nchw_128_128_2_3_1(
		__global const float *filter,
		__global const float *input,
		__global float *output) {
	__local float U[16][128];
	for (int ff = GET_GROUP_ID(); ff < 128; ff += GET_NUM_GROUPS()) {  // out_channel
		for (int rc = GET_LOCAL_ID(); rc < 128; rc += GET_LOCAL_SIZE()) {  // in_channel
			__global const float *g = &filter[((ff * 128) + rc) * 9];  // f(rc, ff)
			U[0][rc] = g[0];
			U[1][rc] = (g[0] + g[1] + g[2])/2;
			U[2][rc] = (g[0] - g[1] + g[2])/2;
			U[3][rc] = g[2];
			U[4][rc] = (g[0] + g[3] + g[6])/2;
			U[5][rc] = (g[0] + g[1] + g[2] + g[3] + g[4] + g[5] + g[6] + g[7] + g[8])/4;
			U[6][rc] = (g[0] - g[1] + g[2] + g[3] - g[4] + g[5] + g[6] - g[7] + g[8])/4;
			U[7][rc] = (g[2] + g[5] + g[8])/2;
			U[8][rc] = (g[0] - g[3] + g[6])/2;
			U[9][rc] = (g[0] + g[1] + g[2] - g[3] - g[4] - g[5] + g[6] + g[7] + g[8])/4;
			U[10][rc] = (g[0] - g[1] + g[2] - g[3] + g[4] - g[5] + g[6] - g[7] + g[8])/4;
			U[11][rc] = (g[2] - g[5] + g[8])/2;
			U[12][rc] = g[6];
			U[13][rc] = (g[6] + g[7] + g[8])/2;
			U[14][rc] = (g[6] - g[7] + g[8])/2;
			U[15][rc] = g[8];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		for (int nn = GET_LOCAL_ID(); nn < 128; nn += GET_LOCAL_SIZE()) {  // batch
			float m[16];
			for (int i = 0; i < 16; ++i) {
				m[i] = 0;
			}

			for (int rc = 0; rc < 128; ++rc) {  // in_channel
				__global const float *d = &input[((nn * 128) + rc) * 4];  // f(nn, rc)
				float V[16];
				V[0] = d[3];
				V[1] = -(d[2] + d[3]);
				V[2] = d[2] - d[3];
				V[3] = d[2];
				V[4] = -(d[1] + d[3]);
				V[5] = d[0] + d[1] + d[2] + d[3];
				V[6] = -(d[0] + d[2]) + d[1] + d[3];
				V[7] = -(d[0] + d[2]);
				V[8] = d[1] - d[3];
				V[9] = -(d[0] + d[1]) + d[2] + d[3];
				V[10] = -(d[1] + d[2]) + d[0] + d[3];
				V[11] = d[0] - d[2];
				V[12] = d[1];
				V[13] = -(d[0] + d[1]);
				V[14] = d[0] - d[1];
				V[15] = d[0];

				m[0] += U[0][rc] * V[0];
				m[1] += U[1][rc] * V[1];
				m[2] += U[2][rc] * V[2];
				m[3] += U[3][rc] * V[3];
				m[4] += U[4][rc] * V[4];
				m[5] += U[5][rc] * V[5];
				m[6] += U[6][rc] * V[6];
				m[7] += U[7][rc] * V[7];
				m[8] += U[8][rc] * V[8];
				m[9] += U[9][rc] * V[9];
				m[10] += U[10][rc] * V[10];
				m[11] += U[11][rc] * V[11];
				m[12] += U[12][rc] * V[12];
				m[13] += U[13][rc] * V[13];
				m[14] += U[14][rc] * V[14];
				m[15] += U[15][rc] * V[15];
			}

			__global float *out = &output[((nn * 128) + ff) * 4];  // f(nn, ff)
			out[0] = m[0] + m[10] + m[1] + m[2] + m[4] + m[5] + m[6] + m[8] + m[9];
			out[1] = -m[10] + m[11] + m[1] - m[2] + m[3] + m[5] - m[6] + m[7] + m[9];
			out[2] = -m[10] + m[12] + m[13] + m[14] + m[4] + m[5] + m[6] - m[8] - m[9];
			out[3] = m[10] - m[11] + m[13] - m[14] + m[15] + m[5] - m[6] + m[7] - m[9];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}
}
