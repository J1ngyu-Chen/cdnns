#include <cdnns_relu.h>
#include <cuda_runtime.h>

#define DEFINE_KERNEL_CDNNS_RELU(SUFFIX, TYPE, FUNCT, OP)                    \
	static __global__ void kernel_cdnns_relu_##SUFFIX(int batch_size,    \
							  int inout_dim,     \
							  const TYPE *input, \
							  TYPE *output)      \
	{                                                                    \
		int io = blockIdx.x * blockDim.x + threadIdx.x;            \
		int b = blockIdx.y * blockDim.y + threadIdx.y;             \
                                                                             \
		if (b < batch_size && io < inout_dim) {                  \
			int idx = b * inout_dim + io;                    \
			TYPE x = input[idx];                                 \
			TYPE y = FUNCT(x);                                   \
			output[idx] OP y;                                    \
		}                                                            \
	}

DEFINE_KERNEL_CDNNS_RELU(ow_fp32, float, CDNNS_FUNCT_RELU_FP32, =)
DEFINE_KERNEL_CDNNS_RELU(ow_fp64, double, CDNNS_FUNCT_RELU_FP64, =)
DEFINE_KERNEL_CDNNS_RELU(accum_fp32, float, CDNNS_FUNCT_RELU_FP32, +=)
DEFINE_KERNEL_CDNNS_RELU(accum_fp64, double, CDNNS_FUNCT_RELU_FP64, +=)

extern "C" {

#define DEFINE_CDNNS_RELU(SUFFIX, TYPE)                              \
	void cdnns_relu_##SUFFIX(int batch_size, int inout_dim,      \
				 const TYPE *input, TYPE *output)    \
	{                                                            \
		dim3 block_dim(16, 16);                              \
		dim3 grid_dim((inout_dim - 1) / block_dim.x + 1,     \
			      (batch_size - 1) / block_dim.y + 1);   \
		kernel_cdnns_relu_##SUFFIX<<<grid_dim, block_dim>>>( \
			batch_size, inout_dim, input, output);       \
	}

DEFINE_CDNNS_RELU(ow_fp32, float)
DEFINE_CDNNS_RELU(ow_fp64, double)
DEFINE_CDNNS_RELU(accum_fp32, float)
DEFINE_CDNNS_RELU(accum_fp64, double)
}