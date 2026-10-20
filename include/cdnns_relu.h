#ifndef CDNNS_RELU_H
#define CDNNS_RELU_H
#include <cdnns.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CDNNS_FUNCT_RELU_FP32(VAR) (VAR > 0.0f) * VAR
#define CDNNS_FUNCT_RELU_FP64(VAR) (VAR > 0.0) * VAR

void cdnns_relu_ow_fp32(int batch_size, int inout_dim,
			const float *input, float *output);

void cdnns_relu_ow_fp64(int batch_size, int inout_dim,
			const double *input, double *output);

void cdnns_relu_accum_fp32(int batch_size, int inout_dim,
			   const float *input, float *output);

void cdnns_relu_accum_fp64(int batch_size, int inout_dim,
			   const double *input,
			   double *output);

#ifdef __cplusplus
}
#endif

#endif
