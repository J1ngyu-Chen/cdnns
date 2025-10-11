#include "core.h"
#if defined(IMPL_BASIC)
void leakyrelu_fwd_default_ow_f32(size_t batch_size, size_t inout_dim,
				  const float *__restrict input,
				  const float *__restrict alpha,
				  float *__restrict output)
{
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t io = 0; io < inout_dim; io++) {
			size_t idx = b * inout_dim + io;
			bool is_greater_than_zero = (input[idx] > 0.0f);
			output[idx] =
				is_greater_than_zero * input[idx] +
				!is_greater_than_zero * (*alpha) * input[idx];
		}
	}
}

void leakyrelu_fwd_default_ow_in_place_f32(size_t batch_size, size_t inout_dim,
					   const float *alpha, float *inout)
{
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t io = 0; io < inout_dim; io++) {
			size_t idx = b * inout_dim + io;
			bool is_greater_than_zero = (inout[idx] > 0.0f);
			inout[idx] =
				is_greater_than_zero * inout[idx] +
				!is_greater_than_zero * (*alpha) * inout[idx];
		}
	}
}

void leakyrelu_fwd_default_accum_f32(size_t batch_size, size_t inout_dim,
				     const float *__restrict input,
				     const float *__restrict alpha,
				     float *__restrict output)
{
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t io = 0; io < inout_dim; io++) {
			size_t idx = b * inout_dim + io;
			bool is_greater_than_zero = (input[idx] > 0.0f);
			output[idx] +=
				is_greater_than_zero * input[idx] +
				!is_greater_than_zero * (*alpha) * input[idx];
		}
	}
}

void leakyrelu_fwd_default_ow_f64(size_t batch_size, size_t inout_dim,
				  const double *__restrict input,
				  const double *__restrict alpha,
				  double *__restrict output)
{
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t io = 0; io < inout_dim; io++) {
			size_t idx = b * inout_dim + io;
			bool is_greater_than_zero = (input[idx] > 0.0);
			output[idx] =
				is_greater_than_zero * input[idx] +
				!is_greater_than_zero * (*alpha) * input[idx];
		}
	}
}

void leakyrelu_fwd_default_ow_in_place_f64(size_t batch_size, size_t inout_dim,
					   const double *alpha, double *inout)
{
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t io = 0; io < inout_dim; io++) {
			size_t idx = b * inout_dim + io;
			bool is_greater_than_zero = (inout[idx] > 0.0);
			inout[idx] =
				is_greater_than_zero * inout[idx] +
				!is_greater_than_zero * (*alpha) * inout[idx];
		}
	}
}

void leakyrelu_fwd_default_accum_f64(size_t batch_size, size_t inout_dim,
				     const double *__restrict input,
				     const double *__restrict alpha,
				     double *__restrict output)
{
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t io = 0; io < inout_dim; io++) {
			size_t idx = b * inout_dim + io;
			bool is_greater_than_zero = (input[idx] > 0.0);
			output[idx] +=
				is_greater_than_zero * input[idx] +
				!is_greater_than_zero * (*alpha) * input[idx];
		}
	}
}
#endif
