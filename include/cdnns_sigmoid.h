#ifndef CDNNS_SIGMOID_H
#define CDNNS_SIGMOID_H
#include <cdnns.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CDNNS_FUNCT_SIGMOID_FP32(VAR) 1.0f / (1.0f + expf(-VAR))
#define CDNNS_FUNCT_SIGMOID_FP64(VAR) 1.0 / (1.0 + exp(-VAR))

void cdnns_sigmoid_ow_fp32(int batch_size, int inout_dim,
			   const float *input, float *output);

void cdnns_sigmoid_ow_fp64(int batch_size, int inout_dim,
			   const double *input,
			   double *output);

void cdnns_sigmoid_accum_fp32(int batch_size, int inout_dim,
			      const float *input,
			      float *output);

void cdnns_sigmoid_accum_fp64(int batch_size, int inout_dim,
			      const double *input,
			      double *output);

#ifdef __cplusplus
}
#endif

#endif
