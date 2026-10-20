#include <cdnns_relu.h>

#define DEFINE_CDNNS_RELU(SUFFIX, TYPE, FUNCT, OP)                \
	void cdnns_relu_##SUFFIX(int batch_size, int inout_dim,   \
				 const TYPE *input, TYPE *output) \
	{                                                         \
		for (int b = 0; b < batch_size; b++) {            \
			for (int io = 0; io < inout_dim; io++) {  \
				int idx = b * inout_dim + io;     \
				TYPE x = input[idx];              \
				TYPE y = FUNCT(x);                \
				output[idx] OP y;                 \
			}                                         \
		}                                                 \
	}

DEFINE_CDNNS_RELU(ow_fp32, float, CDNNS_FUNCT_RELU_FP32, =)
DEFINE_CDNNS_RELU(ow_fp64, double, CDNNS_FUNCT_RELU_FP64, =)
DEFINE_CDNNS_RELU(accum_fp32, float, CDNNS_FUNCT_RELU_FP32, +=)
DEFINE_CDNNS_RELU(accum_fp64, double, CDNNS_FUNCT_RELU_FP64, +=)


