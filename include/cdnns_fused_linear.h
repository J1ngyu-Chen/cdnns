#ifndef CDNNS_FUSED_LINEAR_H
#define CDNNS_FUSED_LINEAR_H
#include <cdnns.h>

#ifdef __cplusplus
extern "C" {
#endif

void cdnns_fused_linear_relu_ow_fp32(int batch_size, int in_dim, int out_dim,
				     const float *RESTRICT input,
				     const float *RESTRICT weights,
				     const float *RESTRICT bias,
				     float *RESTRICT output);

void cdnns_fused_linear_relu_ow_fp64(int batch_size, int in_dim, int out_dim,
				     const double *RESTRICT input,
				     const double *RESTRICT weights,
				     const double *RESTRICT bias,
				     double *RESTRICT output);

void cdnns_fused_linear_relu_accum_fp32(int batch_size, int in_dim, int out_dim,
					const float *RESTRICT input,
					const float *RESTRICT weights,
					const float *RESTRICT bias,
					float *RESTRICT output);

void cdnns_fused_linear_relu_accum_fp64(int batch_size, int in_dim, int out_dim,
					const double *RESTRICT input,
					const double *RESTRICT weights,
					const double *RESTRICT bias,
					double *RESTRICT output);

void cdnns_fused_linear_nobias_relu_ow_fp32(int batch_size, int in_dim,
					     int out_dim,
					     const float *RESTRICT input,
					     const float *RESTRICT weights,
					     float *RESTRICT output);

void cdnns_fused_linear_nobias_relu_ow_fp64(int batch_size, int in_dim,
					     int out_dim,
					     const double *RESTRICT input,
					     const double *RESTRICT weights,
					     double *RESTRICT output);

void cdnns_fused_linear_nobias_relu_accum_fp32(int batch_size, int in_dim,
						int out_dim,
						const float *RESTRICT input,
						const float *RESTRICT weights,
						float *RESTRICT output);

void cdnns_fused_linear_nobias_relu_accum_fp64(int batch_size, int in_dim,
						int out_dim,
						const double *RESTRICT input,
						const double *RESTRICT weights,
						double *RESTRICT output);

#ifdef __cplusplus
}
#endif

#endif
