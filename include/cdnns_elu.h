#ifndef CDNNS_ELU_H
#define CDNNS_ELU_H
#include <cdnns.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CDNNS_FUNCT_ELU_FP32(VAR, ALPHA) \
	(VAR > 0.0f) * VAR + !(VAR > 0.0f) * ALPHA * (expf(VAR) - 1.0f);

#define CDNNS_FUNCT_ELU_FP64(VAR, ALPHA) \
	(VAR > 0.0) * VAR + !(VAR > 0.0) * ALPHA * (exp(VAR) - 1.0);

void cdnns_elu_ow_fp32(int batch_size, int inout_dim, const float *input,
		       float alpha, float *output);

void cdnns_elu_ow_fp64(int batch_size, int inout_dim, const double *input,
		       double alpha, double *output);

void cdnns_elu_accum_fp32(int batch_size, int inout_dim, const float *input,
			  float alpha, float *output);

void cdnns_elu_accum_fp64(int batch_size, int inout_dim, const double *input,
			  double alpha, double *output);

#ifdef __cplusplus
}
#endif

#endif
