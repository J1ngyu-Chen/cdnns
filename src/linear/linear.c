#include <cdnns_linear.h>

#define CDNNS_LINEAR_HEAD(NAME, TYPE)                                       \
	void NAME(int batch_size, int in_dim, int out_dim,                  \
		  const TYPE *RESTRICT input, const TYPE *RESTRICT weights, \
		  const TYPE *RESTRICT bias, TYPE *RESTRICT output)

#define CDNNS_LINEAR_HEAD_NOBIAS(NAME, TYPE)                                \
	void NAME(int batch_size, int in_dim, int out_dim,                  \
		  const TYPE *RESTRICT input, const TYPE *RESTRICT weights, \
		  TYPE *RESTRICT output)

#define CDNNS_LINEAR_BODY(TYPE, INIT, STORE)                           \
	for (int b = 0; b < batch_size; b++) {                         \
		for (int o = 0; o < out_dim; o++) {                    \
			TYPE sum = (TYPE)(INIT);                       \
			for (int i = 0; i < in_dim; i++) {             \
				int idx_in = b * in_dim + i;           \
				int idx_w = o * in_dim + i;            \
				sum += input[idx_in] * weights[idx_w]; \
			}                                              \
			int idx_out = b * out_dim + o;                 \
			output[idx_out] STORE sum;                     \
		}                                                      \
	}

#define DEFINE_CDNNS_LINEAR(HEAD, BODY) \
	HEAD                            \
	{                               \
		BODY                    \
	}

DEFINE_CDNNS_LINEAR(CDNNS_LINEAR_HEAD(cdnns_linear_fp32_ow, float),
		    CDNNS_LINEAR_BODY(float, bias[o], =))

DEFINE_CDNNS_LINEAR(CDNNS_LINEAR_HEAD_NOBIAS(cdnns_linear_fp32_ow_nobias,
					     float),
		    CDNNS_LINEAR_BODY(float, 0.0f, =))

DEFINE_CDNNS_LINEAR(CDNNS_LINEAR_HEAD(cdnns_linear_fp32_accum, float),
		    CDNNS_LINEAR_BODY(float, bias[o], +=))

DEFINE_CDNNS_LINEAR(CDNNS_LINEAR_HEAD_NOBIAS(cdnns_linear_fp32_accum_nobias,
					     float),
		    CDNNS_LINEAR_BODY(float, 0.0f, +=))

DEFINE_CDNNS_LINEAR(CDNNS_LINEAR_HEAD(cdnns_linear_fp64_ow, double),
		    CDNNS_LINEAR_BODY(double, bias[o], =))

DEFINE_CDNNS_LINEAR(CDNNS_LINEAR_HEAD_NOBIAS(cdnns_linear_fp64_ow_nobias,
					     double),
		    CDNNS_LINEAR_BODY(double, 0.0, =))

DEFINE_CDNNS_LINEAR(CDNNS_LINEAR_HEAD(cdnns_linear_fp64_accum, double),
		    CDNNS_LINEAR_BODY(double, bias[o], +=))

DEFINE_CDNNS_LINEAR(CDNNS_LINEAR_HEAD_NOBIAS(cdnns_linear_fp64_accum_nobias,
					     double),
		    CDNNS_LINEAR_BODY(double, 0.0, +=))
