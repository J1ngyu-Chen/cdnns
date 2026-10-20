#ifndef CDNNS_LINEAR_H
#define CDNNS_LINEAR_H
#include <cdnns.h>

#ifdef __cplusplus
extern "C" {
#endif

void cdnns_linear_fp32_ow(int batch_size, int in_dim, int out_dim,
			  const float *RESTRICT input,
			  const float *RESTRICT weights,
			  const float *RESTRICT bias, float *RESTRICT output);

void cdnns_linear_fp32_ow_nobias(int batch_size, int in_dim, int out_dim,
				 const float *RESTRICT input,
				 const float *RESTRICT weights,
				 float *RESTRICT output);

void cdnns_linear_fp32_accum(int batch_size, int in_dim, int out_dim,
			     const float *RESTRICT input,
			     const float *RESTRICT weights,
			     const float *RESTRICT bias,
			     float *RESTRICT output);

void cdnns_linear_fp32_accum_nobias(int batch_size, int in_dim, int out_dim,
				    const float *RESTRICT input,
				    const float *RESTRICT weights,
				    float *RESTRICT output);

void cdnns_linear_fp64_ow(int batch_size, int in_dim, int out_dim,
			  const double *RESTRICT input,
			  const double *RESTRICT weights,
			  const double *RESTRICT bias, double *RESTRICT output);

void cdnns_linear_fp64_ow_nobias(int batch_size, int in_dim, int out_dim,
				 const double *RESTRICT input,
				 const double *RESTRICT weights,
				 double *RESTRICT output);

void cdnns_linear_fp64_accum(int batch_size, int in_dim, int out_dim,
			     const double *RESTRICT input,
			     const double *RESTRICT weights,
			     const double *RESTRICT bias,
			     double *RESTRICT output);

void cdnns_linear_fp64_accum_nobias(int batch_size, int in_dim, int out_dim,
				    const double *RESTRICT input,
				    const double *RESTRICT weights,
				    double *RESTRICT output);

#ifdef __cplusplus
}
#endif

#endif
