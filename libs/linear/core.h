#ifndef CORE_H
#define CORE_H
#include <cdnns.h>

#ifdef __cplusplus
extern "C" {
#endif

void linear_fwd_default_ow_f32(size_t batch_size, size_t in_dim, size_t out_dim,
			       const float *__restrict input,
			       const float *__restrict weight,
			       const float *__restrict bias,
			       float *__restrict output);

void linear_fwd_no_bias_ow_f32(size_t batch_size, size_t in_dim, size_t out_dim,
			       const float *__restrict input,
			       const float *__restrict weight,
			       float *__restrict output);

void linear_fwd_fuse_relu_ow_f32(size_t batch_size, size_t in_dim,
				 size_t out_dim, const float *__restrict input,
				 const float *__restrict weight,
				 const float *__restrict bias,
				 float *__restrict output);

void linear_fwd_default_accum_f32(size_t batch_size, size_t in_dim,
				  size_t out_dim, const float *__restrict input,
				  const float *__restrict weight,
				  const float *__restrict bias,
				  float *__restrict output);

void linear_fwd_no_bias_accum_f32(size_t batch_size, size_t in_dim,
				  size_t out_dim, const float *__restrict input,
				  const float *__restrict weight,
				  float *__restrict output);

void linear_fwd_fuse_relu_accum_f32(size_t batch_size, size_t in_dim,
				    size_t out_dim, const float *__restrict input,
				    const float *__restrict weight,
				    const float *__restrict bias,
				    float *__restrict output);

void linear_fwd_default_ow_f64(size_t batch_size, size_t in_dim, size_t out_dim,
			       const double *__restrict input,
			       const double *__restrict weight,
			       const double *__restrict bias,
			       double *__restrict output);

void linear_fwd_no_bias_ow_f64(size_t batch_size, size_t in_dim, size_t out_dim,
			       const double *__restrict input,
			       const double *__restrict weight,
			       double *__restrict output);

void linear_fwd_fuse_relu_ow_f64(size_t batch_size, size_t in_dim,
				 size_t out_dim, const double *__restrict input,
				 const double *__restrict weight,
				 const double *__restrict bias,
				 double *__restrict output);

void linear_fwd_default_accum_f64(size_t batch_size, size_t in_dim,
				  size_t out_dim, const double *__restrict input,
				  const double *__restrict weight,
				  const double *__restrict bias,
				  double *__restrict output);

void linear_fwd_no_bias_accum_f64(size_t batch_size, size_t in_dim,
				  size_t out_dim, const double *__restrict input,
				  const double *__restrict weight,
				  double *__restrict output);

void linear_fwd_fuse_relu_accum_f64(size_t batch_size, size_t in_dim,
				    size_t out_dim,
				    const double *__restrict input,
				    const double *__restrict weight,
				    const double *__restrict bias,
				    double *__restrict output);

#ifdef __cplusplus
}
#endif

#endif
