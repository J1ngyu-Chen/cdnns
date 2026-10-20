#ifndef CDNNS_TANH_H
#define CDNNS_TANH_H
#include <cdnns.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CDNNS_FUNCT_TANH_FP32(VAR) (expf(VAR) - expf(-VAR)) / (expf(VAR) + expf(-VAR));
#define CDNNS_FUNCT_TANH_FP64(VAR) (exp(VAR) - exp(-VAR)) / (exp(VAR) + exp(-VAR));

void cdnns_tanh_ow_fp32(int batch_size, int inout_dim,
		       const float *RESTRICT input, float *RESTRICT output);

void cdnns_tanh_ow_fp64(int batch_size, int inout_dim,
		       const double *RESTRICT input, double *RESTRICT output);

void cdnns_tanh_accum_fp32(int batch_size, int inout_dim,
			  const float *RESTRICT input, float *RESTRICT output);

void cdnns_tanh_accum_fp64(int batch_size, int inout_dim,
			  const double *RESTRICT input,
			  double *RESTRICT output);

#ifdef __cplusplus
}
#endif

#endif
