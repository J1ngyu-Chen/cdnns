#ifndef CDNNS_LEAKYRELU_H
#define CDNNS_LEAKYRELU_H
#include <cdnns.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CDNNS_FUNCT_LEAKYRELU_FP32(VAR, ALPHA) \
	(VAR > 0.0f) * VAR + !(VAR > 0.0f) * alpha * VAR;

#define CDNNS_FUNCT_LEAKYRELU_FP64(VAR, ALPHA) \
	(VAR > 0.0) * VAR + !(VAR > 0.0) * alpha * VAR;

void cdnns_leakyrelu_ow_fp32(int batch_size, int inout_dim, const float *input,
			     float alpha, float *output);

void cdnns_leakyrelu_ow_fp64(int batch_size, int inout_dim, const double *input,
			     double alpha, double *output);

void cdnns_leakyrelu_accum_fp32(int batch_size, int inout_dim,
				const float *input, float alpha, float *output);

void cdnns_leakyrelu_accum_fp64(int batch_size, int inout_dim,
				const double *input, double alpha,
				double *output);

#ifdef __cplusplus
}
#endif

#endif
