#include <cdnns_linear.h>
#include <cuda_runtime.h>
#include <stddef.h>

#define DEFINE_KERNEL_CDNNS_LINEAR(SUFFIX, TYPE, INIT, OP)                \
	static __global__ void kernel_cdnns_linear_##SUFFIX(              \
		int batch_size, int in_dim, int out_dim,                  \
		const TYPE *RESTRICT input, const TYPE *RESTRICT weights, \
		const TYPE *RESTRICT bias, TYPE *RESTRICT output)         \
	{                                                                 \
		int o = blockIdx.x * blockDim.x + threadIdx.x;            \
		int b = blockIdx.y * blockDim.y + threadIdx.y;            \
		if (b < batch_size && o < out_dim) {                      \
			TYPE sum = (TYPE)(INIT);                          \
			for (int i = 0; i < in_dim; i++) {                \
				int idx_in = b * in_dim + i;              \
				int idx_w = o * in_dim + i;               \
				sum += input[idx_in] * weights[idx_w];    \
			}                                                 \
			int idx_out = b * out_dim + o;                    \
			output[idx_out] OP sum;                           \
		}                                                         \
	}

DEFINE_KERNEL_CDNNS_LINEAR(ow_fp32, float, bias[o], =)
DEFINE_KERNEL_CDNNS_LINEAR(ow_fp64, double, bias[o], =)
DEFINE_KERNEL_CDNNS_LINEAR(accum_fp32, float, bias[o], +=)
DEFINE_KERNEL_CDNNS_LINEAR(accum_fp64, double, bias[o], +=)
DEFINE_KERNEL_CDNNS_LINEAR(nobias_ow_fp32, float, 0, =)
DEFINE_KERNEL_CDNNS_LINEAR(nobias_ow_fp64, double, 0, =)
DEFINE_KERNEL_CDNNS_LINEAR(nobias_accum_fp32, float, 0, +=)
DEFINE_KERNEL_CDNNS_LINEAR(nobias_accum_fp64, double, 0, +=)

extern "C" {
#define CDNNS_LINEAR_BODY(SUFFIX, BIAS)                        \
	dim3 block_dim(16, 16);                                \
	dim3 grid_dim((out_dim - 1) / block_dim.x + 1,         \
		      (batch_size - 1) / block_dim.y + 1);     \
	kernel_cdnns_linear_##SUFFIX<<<grid_dim, block_dim>>>( \
		batch_size, in_dim, out_dim, input, weights, BIAS, output);

#define DEFINE_CDNNS_LINEAR(SUFFIX, TYPE)                                   \
	void cdnns_linear_##SUFFIX(int batch_size, int in_dim, int out_dim, \
				   const TYPE *RESTRICT input,              \
				   const TYPE *RESTRICT weights,            \
				   const TYPE *RESTRICT bias,               \
				   TYPE *RESTRICT output)                   \
	{                                                                   \
		CDNNS_LINEAR_BODY(SUFFIX, bias)                             \
	}

#define DEFINE_CDNNS_LINEAR_NOBIAS(SUFFIX, TYPE)                        \
	void cdnns_linear_nobias_##SUFFIX(int batch_size, int in_dim,   \
					  int out_dim,                  \
					  const TYPE *RESTRICT input,   \
					  const TYPE *RESTRICT weights, \
					  TYPE *RESTRICT output)        \
	{                                                               \
		CDNNS_LINEAR_BODY(SUFFIX, NULL)                         \
	}

DEFINE_CDNNS_LINEAR(ow_fp32, float)
DEFINE_CDNNS_LINEAR(ow_fp64, double)
DEFINE_CDNNS_LINEAR(accum_fp32, float)
DEFINE_CDNNS_LINEAR(accum_fp64, double)
DEFINE_CDNNS_LINEAR_NOBIAS(ow_fp32, float)
DEFINE_CDNNS_LINEAR_NOBIAS(ow_fp64, double)
DEFINE_CDNNS_LINEAR_NOBIAS(accum_fp32, float)
DEFINE_CDNNS_LINEAR_NOBIAS(accum_fp64, double)
}