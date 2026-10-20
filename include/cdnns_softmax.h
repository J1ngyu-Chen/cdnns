#ifndef CDNNS_SOFTMAX_H
#define CDNNS_SOFTMAX_H
#include <cdnns.h>

#ifdef __cplusplus
extern "C" {
#endif

void cdnns_softmax_ow_fp32(int batch_size, int inout_dim,
			   const float *RESTRICT input, float *RESTRICT output);

void cdnns_softmax_ow_fp64(int batch_size, int inout_dim,
			   const double *RESTRICT input,
			   double *RESTRICT output);

void cdnns_softmax_accum_fp64(int batch_size, int inout_dim,
			      const double *RESTRICT input,
			      double *RESTRICT output);

void cdnns_softmax_accum_fp32(int batch_size, int inout_dim,
			      const float *RESTRICT input,
			      float *RESTRICT output);

#ifdef __cplusplus
}
#endif

#endif
