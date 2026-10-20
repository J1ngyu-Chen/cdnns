#include "cdnns_elu.h"
#include <math.h>
#include <cdnns_fused_linear.h>

#define CDNNS_FUSED_LINEAR_HEAD(NAME, TYPE)                                \
	void NAME(int batch_size, int in_dim, int out_dim,                 \
		  const TYPE *RESTRICT input, const TYPE *RESTRICT weight, \
		  const TYPE *RESTRICT bias, TYPE *RESTRICT output)

#define CDNNS_FUSED_LINEAR_HEAD_NOBIAS(NAME, TYPE)                         \
	void NAME(int batch_size, int in_dim, int out_dim,                 \
		  const TYPE *RESTRICT input, const TYPE *RESTRICT weight, \
		  TYPE *RESTRICT output)

#define CDNNS_FUSED_LINEAR_HEAD_ARGS(NAME, TYPE, ARGS)                     \
	void NAME(int batch_size, int in_dim, int out_dim,                 \
		  const TYPE *RESTRICT input, const TYPE *RESTRICT weight, \
		  const TYPE *RESTRICT bias, ARGS, TYPE *RESTRICT output)

#define CDNNS_FUSED_LINEAR_HEAD_NOBIAS_ARGS(NAME, TYPE, ARGS)              \
	void NAME(int batch_size, int in_dim, int out_dim,                 \
		  const TYPE *RESTRICT input, const TYPE *RESTRICT weight, \
		  ARGS, TYPE *RESTRICT output)

#define CDNNS_FUSED_LINEAR_BODY(TYPE, INIT, FUNCT, STORE)             \
	for (int b = 0; b < batch_size; b++) {                        \
		for (int o = 0; o < out_dim; o++) {                   \
			TYPE sum = (TYPE)(INIT);                      \
			for (int i = 0; i < in_dim; i++) {            \
				int idx_in = b * in_dim + i;          \
				int idx_w = o * in_dim + i;           \
				sum += input[idx_in] * weight[idx_w]; \
			}                                             \
			sum = FUNCT(sum);                             \
			int idx_out = b * out_dim + o;                \
			output[idx_out] STORE sum;                    \
		}                                                     \
	}

#define CDNNS_FUSED_LINEAR_BODY_ARGS(TYPE, INIT, FUNCT, ARGS, STORE)  \
	for (int b = 0; b < batch_size; b++) {                        \
		for (int o = 0; o < out_dim; o++) {                   \
			TYPE sum = (TYPE)(INIT);                      \
			for (int i = 0; i < in_dim; i++) {            \
				int idx_in = b * in_dim + i;          \
				int idx_w = o * in_dim + i;           \
				sum += input[idx_in] * weight[idx_w]; \
			}                                             \
			sum = FUNCT(sum, ARGS);                       \
			int idx_out = b * out_dim + o;                \
			output[idx_out] STORE sum;                    \
		}                                                     \
	}

#define DEFINE_CDNNS_FUSED_LINEAR(HEAD, BODY) \
	HEAD                                  \
	{                                     \
		BODY                          \
	}

DEFINE_CDNNS_FUSED_LINEAR(CDNNS_FUSED_LINEAR_HEAD_ARGS(cdnns_linear_fp32_ow,
						       float, float alpha),
			  CDNNS_FUSED_LINEAR_BODY_ARGS(float, bias[o],
						       CDNNS_FUNCT_ELU_FP32,
						       alpha, =))

DEFINE_CDNNS_FUSED_LINEAR(
	CDNNS_FUSED_LINEAR_HEAD_NOBIAS(cdnns_linear_fp32_ow_nobias, float),
	CDNNS_FUSED_LINEAR_BODY(float, 0.0f, =))

DEFINE_CDNNS_FUSED_LINEAR(CDNNS_FUSED_LINEAR_HEAD(cdnns_linear_fp32_accum,
						  float),
			  CDNNS_FUSED_LINEAR_BODY(float, bias[o], +=))

DEFINE_CDNNS_FUSED_LINEAR(
	CDNNS_LINEAR_HEAD_NOBIAS(cdnns_linear_fp32_accum_nobias, float),
	CDNNS_LINEAR_BODY(float, 0.0f, +=))

DEFINE_CDNNS_FUSED_LINEAR(CDNNS_LINEAR_HEAD(cdnns_linear_fp64_ow, double),
			  CDNNS_LINEAR_BODY(double, bias[o], =))

DEFINE_CDNNS_FUSED_LINEAR(CDNNS_LINEAR_HEAD_NOBIAS(cdnns_linear_fp64_ow_nobias,
						   double),
			  CDNNS_LINEAR_BODY(double, 0.0, =))

DEFINE_CDNNS_FUSED_LINEAR(CDNNS_LINEAR_HEAD(cdnns_linear_fp64_accum, double),
			  CDNNS_LINEAR_BODY(double, bias[o], +=))

DEFINE_CDNNS_FUSED_LINEAR(
	CDNNS_LINEAR_HEAD_NOBIAS(cdnns_linear_fp64_accum_nobias, double),
	CDNNS_LINEAR_BODY(double, 0.0, +=))
