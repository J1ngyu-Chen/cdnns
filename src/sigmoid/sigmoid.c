#include <cdnns_sigmoid.h>
#include <math.h>

#define DEFINE_CDNNS_SIGMOID(SUFFIX, TYPE, FUNCT, OP)                \
	void cdnns_sigmoid_##SUFFIX(int batch_size, int inout_dim,   \
				    const TYPE *input, TYPE *output) \
	{                                                            \
		for (int b = 0; b < batch_size; b++) {               \
			for (int io = 0; io < inout_dim; io++) {     \
				int idx = b * inout_dim + io;        \
				TYPE x = input[idx];                 \
				TYPE y = FUNCT(x);                   \
				output[idx] OP y;                    \
			}                                            \
		}                                                    \
	}

DEFINE_CDNNS_SIGMOID(ow_fp32, float, CDNNS_FUNCT_SIGMOID_FP32, =)
DEFINE_CDNNS_SIGMOID(ow_fp64, double, CDNNS_FUNCT_SIGMOID_FP64, =)
DEFINE_CDNNS_SIGMOID(accum_fp32, float, CDNNS_FUNCT_SIGMOID_FP32, +=)
DEFINE_CDNNS_SIGMOID(accum_fp64, double, CDNNS_FUNCT_SIGMOID_FP64, +=)
