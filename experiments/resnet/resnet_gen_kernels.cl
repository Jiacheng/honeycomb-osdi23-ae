#define NUM_CH_PER_WG 1
#define NUM_IM_BLKS_X 1
#define NUM_IM_BLKS 1
#define LOCAL_MEM_SIZE 1105
#define STRIDE_GT_1 1
#define TILE_SZ_X 32
#define TILE_SZ_Y 8
#define USE_IM_OFF_GUARD 1
#define MIOPEN_USE_FP32 1
#define MIOPEN_USE_RNE_BFLOAT16 1

#define IM2D2COL_FUNC_NAME Im2d2Col_wg1_x1_imb1_1105_stride1
#include "MIOpenIm2d2Col.inc"

#undef NUM_IM_BLKS
#define NUM_IM_BLKS 2
#define IM2D2COL_FUNC_NAME Im2d2Col_wg1_x1_imb2_1105_stride1
#include "MIOpenIm2d2Col.inc"

#undef NUM_IM_BLKS
#define NUM_IM_BLKS 4
#define IM2D2COL_FUNC_NAME Im2d2Col_wg1_x1_imb4_1105_stride1
#include "MIOpenIm2d2Col.inc"

#undef NUM_IM_BLKS_X
#undef NUM_IM_BLKS
#undef LOCAL_MEM_SIZE
#define NUM_IM_BLKS_X 4
#define NUM_IM_BLKS 56
#define LOCAL_MEM_SIZE 1449
#define IM2D2COL_FUNC_NAME Im2d2Col_wg1_x4_imb56_1449_stride1
#include "MIOpenIm2d2Col.inc"

#undef NUM_CH_PER_WG
#undef NUM_IM_BLKS_X
#undef NUM_IM_BLKS
#undef LOCAL_MEM_SIZE
#undef STRIDE_GT_1
#define NUM_CH_PER_WG 4
#define NUM_IM_BLKS_X 1
#define NUM_IM_BLKS 1
#define LOCAL_MEM_SIZE 544
#define STRIDE_GT_1 0
#define IM2D2COL_FUNC_NAME Im2d2Col_wg4_x1_imb1_544_stride0
#include "MIOpenIm2d2Col.inc"

#undef NUM_CH_PER_WG
#undef NUM_IM_BLKS_X
#undef NUM_IM_BLKS
#undef LOCAL_MEM_SIZE
#undef STRIDE_GT_1
#undef TILE_SZ_X
#undef TILE_SZ_Y
#undef USE_IM_OFF_GUARD
#undef MIOPEN_USE_FP32
#undef MIOPEN_USE_RNE_BFLOAT16

#define MIOPEN_USE_FP32 1
#include "MIOpenUtilKernels4.inc"