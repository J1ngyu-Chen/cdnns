#include <cdnns_softmax.h>
#include <cuda_runtime.h>

#define DEFINE_KERNEL_CDNNS_SOFTMAX(SUFFIX, TYPE, OP, EXP)        \
	static __global__ void kernel_cdnns_softmax_##SUFFIX(     \
		int batch_size, int inout_dim, const TYPE *input, \
		TYPE *output)                                     \
	{                                                         \
		int io = blockIdx.x * blockDim.x + threadIdx.x;   \
		int b = blockIdx.y * blockDim.y + threadIdx.y;    \
		if (b < batch_size && io < inout_dim) {           \
			TYPE sum = (TYPE)0;                       \
			for (int io = 0; io < inout_dim; io++) {  \
				int idx = b * inout_dim + io;     \
				TYPE x = input[idx];              \
				sum += EXP(x);                    \
			}                                         \
			int idx = b * inout_dim + io;             \
			TYPE x = input[idx];                      \
			TYPE y = EXP(x) / sum;                    \
			output[idx] OP y;                         \
		}                                                 \
	}

DEFINE_KERNEL_CDNNS_SOFTMAX(ow_fp32, float, =, expf)
DEFINE_KERNEL_CDNNS_SOFTMAX(ow_fp64, double, =, exp)
DEFINE_KERNEL_CDNNS_SOFTMAX(accum_fp32, float, +=, expf)
DEFINE_KERNEL_CDNNS_SOFTMAX(accum_fp64, double, +=, exp)

extern "C" {
#define DEFINE_CDNNS_SOFTMAX(SUFFIX, TYPE)                              \
	void cdnns_softmax_##SUFFIX(int batch_size, int inout_dim,      \
				    const TYPE *input, TYPE *output)    \
	{                                                               \
		dim3 block_dim(16, 16);                                 \
		dim3 grid_dim((inout_dim - 1) / block_dim.x + 1,        \
			      (batch_size - 1) / block_dim.y + 1);      \
		kernel_cdnns_softmax_##SUFFIX<<<grid_dim, block_dim>>>( \
			batch_size, inout_dim, input, output);          \
	}

DEFINE_CDNNS_SOFTMAX(ow_fp32, float)
DEFINE_CDNNS_SOFTMAX(ow_fp64, double)
DEFINE_CDNNS_SOFTMAX(accum_fp32, float)
DEFINE_CDNNS_SOFTMAX(accum_fp64, double)
}