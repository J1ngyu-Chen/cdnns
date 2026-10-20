#include <cdnns_softmax.h>
#include <math.h>

#define CDNNS_FUNCT_SOFTMAX_FP32()

#define DEFINE_CDNNS_SOFTMAX(SUFFIX, OP, TYPE, EXP)                  \
	void cdnns_softmax_##SUFFIX(int batch_size, int inout_dim,   \
				    const TYPE *input, TYPE *output) \
	{                                                            \
		for (int b = 0; b < batch_size; b++) {               \
			TYPE sum = (TYPE)0;                          \
			for (int io = 0; io < inout_dim; io++) {     \
				int idx = b * inout_dim + io;        \
				TYPE x = input[idx];                 \
				sum += EXP(x);                       \
			}                                            \
			for (int io = 0; io < inout_dim; io++) {     \
				int idx = b * inout_dim + io;        \
				TYPE x = input[idx];                 \
				TYPE y = EXP(x) / sum;               \
				output[idx] OP y;                    \
			}                                            \
		}                                                    \
	}

DEFINE_CDNNS_SOFTMAX(ow_fp32, =, float, expf)
DEFINE_CDNNS_SOFTMAX(ow_fp64, =, double, exp)
DEFINE_CDNNS_SOFTMAX(accum_fp32, +=, float, expf)
DEFINE_CDNNS_SOFTMAX(accum_fp64, +=, double, exp)
