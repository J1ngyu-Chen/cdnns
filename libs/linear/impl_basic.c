#include "core.h"
#if defined(IMPL_BASIC)
void linear_fwd_default_ow_f32(size_t batch_size, size_t in_dim, size_t out_dim,
			       const float *__restrict input,
			       const float *__restrict weight,
			       const float *__restrict bias,
			       float *__restrict output)
{
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t o = 0; o < out_dim; o++) {
			float sum = bias[o];
			for (size_t i = 0; i < in_dim; i++) {
				sum += input[b * in_dim + i] *
				       weight[o * in_dim + i];
			}
			output[b * out_dim + o] = sum;
		}
	}
}

void linear_fwd_no_bias_ow_f32(size_t batch_size, size_t in_dim, size_t out_dim,
			       const float *__restrict input,
			       const float *__restrict weight,
			       float *__restrict output)
{
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t o = 0; o < out_dim; o++) {
			float sum = 0.0f;
			for (size_t i = 0; i < in_dim; i++) {
				sum += input[b * in_dim + i] *
				       weight[o * in_dim + i];
			}
			output[b * out_dim + o] = sum;
		}
	}
}

void linear_fwd_fuse_relu_ow_f32(size_t batch_size, size_t in_dim,
				 size_t out_dim, const float *__restrict input,
				 const float *__restrict weight,
				 const float *__restrict bias,
				 float *__restrict output)
{
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t o = 0; o < out_dim; o++) {
			float sum = bias[o];
			for (size_t i = 0; i < in_dim; i++) {
				sum += input[b * in_dim + i] *
				       weight[o * in_dim + i];
			}
			sum = (sum > 0.0f) * sum;
			output[b * out_dim + o] = sum;
		}
	}
}

void linear_fwd_default_accum_f32(size_t batch_size, size_t in_dim,
				  size_t out_dim, const float *__restrict input,
				  const float *__restrict weight,
				  const float *__restrict bias,
				  float *__restrict output)
{
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t o = 0; o < out_dim; o++) {
			float sum = bias[o];
			for (size_t i = 0; i < in_dim; i++) {
				sum += input[b * in_dim + i] *
				       weight[o * in_dim + i];
			}
			output[b * out_dim + o] += sum;
		}
	}
}

void linear_fwd_no_bias_accum_f32(size_t batch_size, size_t in_dim,
				  size_t out_dim, const float *__restrict input,
				  const float *__restrict weight,
				  float *__restrict output)
{
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t o = 0; o < out_dim; o++) {
			float sum = 0.0f;
			for (size_t i = 0; i < in_dim; i++) {
				sum += input[b * in_dim + i] *
				       weight[o * in_dim + i];
			}
			output[b * out_dim + o] = sum;
		}
	}
}

void linear_fwd_fuse_relu_accum_f32(size_t batch_size, size_t in_dim,
				    size_t out_dim,
				    const float *__restrict input,
				    const float *__restrict weight,
				    const float *__restrict bias,
				    float *__restrict output)
{
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t o = 0; o < out_dim; o++) {
			float sum = bias[o];
			for (size_t i = 0; i < in_dim; i++) {
				sum += input[b * in_dim + i] *
				       weight[o * in_dim + i];
			}
			sum = (sum > 0.0f) * sum;
			output[b * out_dim + o] += sum;
		}
	}
}

void linear_fwd_default_ow_f64(size_t batch_size, size_t in_dim, size_t out_dim,
			       const double *__restrict input,
			       const double *__restrict weight,
			       const double *__restrict bias,
			       double *__restrict output)
{
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t o = 0; o < out_dim; o++) {
			double sum = bias[o];
			for (size_t i = 0; i < in_dim; i++) {
				sum += input[b * in_dim + i] *
				       weight[o * in_dim + i];
			}
			output[b * out_dim + o] = sum;
		}
	}
}

void linear_fwd_no_bias_ow_f64(size_t batch_size, size_t in_dim, size_t out_dim,
			       const double *__restrict input,
			       const double *__restrict weight,
			       double *__restrict output)
{
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t o = 0; o < out_dim; o++) {
			double sum = 0.0;
			for (size_t i = 0; i < in_dim; i++) {
				sum += input[b * in_dim + i] *
				       weight[o * in_dim + i];
			}
			output[b * out_dim + o] = sum;
		}
	}
}

void linear_fwd_fuse_relu_ow_f64(size_t batch_size, size_t in_dim,
				 size_t out_dim, const double *__restrict input,
				 const double *__restrict weight,
				 const double *__restrict bias,
				 double *__restrict output)
{
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t o = 0; o < out_dim; o++) {
			double sum = bias[o];
			for (size_t i = 0; i < in_dim; i++) {
				sum += input[b * in_dim + i] *
				       weight[o * in_dim + i];
			}
			sum = (sum > 0.0) * sum;
			output[b * out_dim + o] = sum;
		}
	}
}

void linear_fwd_default_accum_f64(size_t batch_size, size_t in_dim,
				  size_t out_dim,
				  const double *__restrict input,
				  const double *__restrict weight,
				  const double *__restrict bias,
				  double *__restrict output)
{
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t o = 0; o < out_dim; o++) {
			double sum = bias[o];
			for (size_t i = 0; i < in_dim; i++) {
				sum += input[b * in_dim + i] *
				       weight[o * in_dim + i];
			}
			output[b * out_dim + o] += sum;
		}
	}
}

void linear_fwd_no_bias_accum_f64(size_t batch_size, size_t in_dim,
				  size_t out_dim,
				  const double *__restrict input,
				  const double *__restrict weight,
				  double *__restrict output)
{
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t o = 0; o < out_dim; o++) {
			double sum = 0.0;
			for (size_t i = 0; i < in_dim; i++) {
				sum += input[b * in_dim + i] *
				       weight[o * in_dim + i];
			}
			output[b * out_dim + o] += sum;
		}
	}
}

void linear_fwd_fuse_relu_accum_f64(size_t batch_size, size_t in_dim,
				    size_t out_dim,
				    const double *__restrict input,
				    const double *__restrict weight,
				    const double *__restrict bias,
				    double *__restrict output)
{
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t o = 0; o < out_dim; o++) {
			double sum = bias[o];
			for (size_t i = 0; i < in_dim; i++) {
				sum += input[b * in_dim + i] *
				       weight[o * in_dim + i];
			}
			sum = (sum > 0.0) * sum;
			output[b * out_dim + o] += sum;
		}
	}
}
#endif
