#ifndef CORE_H
#define CORE_H
#include <cdnns.h>

#ifdef __cplusplus
extern "C" {
#endif

void sigmoid_fwd_default_ow_f32(size_t batch_size, size_t inout_dim,
				const float *__restrict input,
				float *__restrict output);

void sigmoid_fwd_default_ow_in_place_f32(size_t batch_size, size_t inout_dim,
					 float *inout);

void sigmoid_fwd_default_accum_f32(size_t batch_size, size_t inout_dim,
				   const float *__restrict input,
				   float *__restrict output);

void sigmoid_fwd_default_ow_f64(size_t batch_size, size_t inout_dim,
				const double *__restrict input,
				double *__restrict output);

void sigmoid_fwd_default_ow_in_place_f64(size_t batch_size, size_t inout_dim,
					 double *inout);

void sigmoid_fwd_default_accum_f64(size_t batch_size, size_t inout_dim,
				   const double *__restrict input,
				   double *__restrict output);

#ifdef __cplusplus
}
#endif

#endif
