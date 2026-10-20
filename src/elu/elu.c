#include <cdnns_elu.h>
#include <stdbool.h>
#include <math.h>

#define DEFINE_CDNNS_ELU(SUFFIX, TYPE, FUNCT, OP)                    \
	void cdnns_elu_##SUFFIX(int batch_size, int inout_dim, \
				      const TYPE *input, TYPE alpha, \
				      TYPE *output)                  \
	{                                                            \
		for (int b = 0; b < batch_size; b++) {               \
			for (int io = 0; io < inout_dim; io++) {     \
				int idx = b * inout_dim + io;        \
				TYPE x = input[idx];                 \
				TYPE y = FUNCT(x, alpha);            \
				output[idx] OP y;                    \
			}                                            \
		}                                                    \
	}

DEFINE_CDNNS_ELU(ow_fp32, float, CDNNS_FUNCT_ELU_FP32, =)
DEFINE_CDNNS_ELU(ow_fp64, double, CDNNS_FUNCT_ELU_FP64, =)
DEFINE_CDNNS_ELU(accum_fp32, float, CDNNS_FUNCT_ELU_FP32, +=)
DEFINE_CDNNS_ELU(accum_fp64, double, CDNNS_FUNCT_ELU_FP64, +=)
