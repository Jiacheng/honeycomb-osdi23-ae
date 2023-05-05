/*
 * winograd.h
 *
 *  Created on: 2022Äê8ÔÂ6ÈÕ
 *      Author: Xing
 */

#ifndef EXPERIMENTS_RESNET_WINOGRAD_H_
#define EXPERIMENTS_RESNET_WINOGRAD_H_

static void winograd_conv2d_nchw_128_128_2_3_1_init(float *output) {
	for (int nn = 0; nn < 128; ++nn) {  // batch
		for (int ff = 0; ff < 128; ++ff) {  // out_channel
			float *out = &output[((nn * 128) + ff) * 4];  // f(nn, ff)
			for (int i = 0; i < 4; ++i) {
				out[i] = 0;
			}
		}
	}
}

static void winograd_conv2d_nchw_128_128_2_3_1(const float *input, const float *filter, float *output) {
	float U[16][128];
	for (int ff = 0; ff < 128; ++ff) {  // out_channel
		for (int rc = 0; rc < 128; ++rc) {  // in_channel
			const float *g = &filter[((ff * 128) + rc) * 9];  // f(rc, ff)
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

		for (int nn = 0; nn < 128; ++nn) {  // batch
			float m[16];
			for (int i = 0; i < 16; ++i) {
				m[i] = 0;
			}

			for (int rc = 0; rc < 128; ++rc) {  // in_channel
				const float *d = &input[((nn * 128) + rc) * 4];  // f(nn, rc)
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

			float *out = &output[((nn * 128) + ff) * 4];  // f(nn, ff)
			out[0] = m[0] + m[10] + m[1] + m[2] + m[4] + m[5] + m[6] + m[8] + m[9];
			out[1] = -m[10] + m[11] + m[1] - m[2] + m[3] + m[5] - m[6] + m[7] + m[9];
			out[2] = -m[10] + m[12] + m[13] + m[14] + m[4] + m[5] + m[6] - m[8] - m[9];
			out[3] = m[10] - m[11] + m[13] - m[14] + m[15] + m[5] - m[6] + m[7] - m[9];
		}
	}
}


#endif /* EXPERIMENTS_RESNET_WINOGRAD_H_ */
