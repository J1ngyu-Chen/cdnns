#include "core.h"
#if defined(IMPL_CUDA)
#include <cuda_runtime.h>

static __global__ void
kernel_tanh_fwd_default_ow_f32(size_t batch_size, size_t inout_dim,
			       const float *__restrict input,
			       float *__restrict output);

static __global__ void
kernel_tanh_fwd_default_ow_in_place_f32(size_t batch_size, size_t inout_dim,
					float *inout);

static __global__ void
kernel_tanh_fwd_default_accum_f32(size_t batch_size, size_t inout_dim,
				  const float *__restrict input,
				  float *__restrict output);

static __global__ void
kernel_tanh_fwd_default_ow_f64(size_t batch_size, size_t inout_dim,
			       const double *__restrict input,
			       double *__restrict output);

static __global__ void
kernel_tanh_fwd_default_ow_in_place_f64(size_t batch_size, size_t inout_dim,
					double *inout);

static __global__ void
kernel_tanh_fwd_default_accum_f64(size_t batch_size, size_t inout_dim,
				  const double *__restrict input,
				  double *__restrict output);

extern "C" {
void tanh_fwd_default_ow_f32(size_t batch_size, size_t inout_dim,
			     const float *__restrict input,
			     float *__restrict output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((inout_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_tanh_fwd_default_ow_f32<<<grid_dim, block_dim>>>(
		batch_size, inout_dim, input, output);
}

void tanh_fwd_default_ow_in_place_f32(size_t batch_size, size_t inout_dim,
				      float *inout)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((inout_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_tanh_fwd_default_ow_in_place_f32<<<grid_dim, block_dim>>>(
		batch_size, inout_dim, inout);
}

void tanh_fwd_default_accum_f32(size_t batch_size, size_t inout_dim,
				const float *__restrict input,
				float *__restrict output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((inout_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_tanh_fwd_default_accum_f32<<<grid_dim, block_dim>>>(
		batch_size, inout_dim, input, output);
}

void tanh_fwd_default_ow_f64(size_t batch_size, size_t inout_dim,
			     const double *__restrict input,
			     double *__restrict output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((inout_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_tanh_fwd_default_ow_f64<<<grid_dim, block_dim>>>(
		batch_size, inout_dim, input, output);
}

void tanh_fwd_default_ow_in_place_f64(size_t batch_size, size_t inout_dim,
				      double *inout)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((inout_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_tanh_fwd_default_ow_in_place_f64<<<grid_dim, block_dim>>>(
		batch_size, inout_dim, inout);
}

void tanh_fwd_default_accum_f64(size_t batch_size, size_t inout_dim,
				const double *__restrict input,
				double *__restrict output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((inout_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_tanh_fwd_default_accum_f64<<<grid_dim, block_dim>>>(
		batch_size, inout_dim, input, output);
}
}

static __global__ void
kernel_tanh_fwd_default_ow_f32(size_t batch_size, size_t inout_dim,
			       const float *__restrict input,
			       float *__restrict output)
{
	size_t k_io = blockIdx.x * blockDim.x + threadIdx.x;
	size_t k_b = blockIdx.y * blockDim.y + threadIdx.y;

	if (k_b < batch_size && k_io < inout_dim) {
		size_t idx = k_b * inout_dim + k_io;
		float exp_p = expf(input[idx]);
		float exp_n = expf(-input[idx]);
		output[idx] = (exp_p - exp_n) / (exp_p + exp_n);
	}
}

static __global__ void
kernel_tanh_fwd_default_ow_in_place_f32(size_t batch_size, size_t inout_dim,
					float *inout)
{
	size_t k_io = blockIdx.x * blockDim.x + threadIdx.x;
	size_t k_b = blockIdx.y * blockDim.y + threadIdx.y;

	if (k_b < batch_size && k_io < inout_dim) {
		size_t idx = k_b * inout_dim + k_io;
		float exp_p = expf(inout[idx]);
		float exp_n = expf(-inout[idx]);
		inout[idx] = (exp_p - exp_n) / (exp_p + exp_n);
	}
}

static __global__ void
kernel_tanh_fwd_default_accum_f32(size_t batch_size, size_t inout_dim,
				  const float *__restrict input,
				  float *__restrict output)
{
	size_t k_io = blockIdx.x * blockDim.x + threadIdx.x;
	size_t k_b = blockIdx.y * blockDim.y + threadIdx.y;

	if (k_b < batch_size && k_io < inout_dim) {
		size_t idx = k_b * inout_dim + k_io;
		float exp_p = expf(input[idx]);
		float exp_n = expf(-input[idx]);
		output[idx] += (exp_p - exp_n) / (exp_p + exp_n);
	}
}

static __global__ void
kernel_tanh_fwd_default_ow_f64(size_t batch_size, size_t inout_dim,
			       const double *__restrict input,
			       double *__restrict output)
{
	size_t k_io = blockIdx.x * blockDim.x + threadIdx.x;
	size_t k_b = blockIdx.y * blockDim.y + threadIdx.y;

	if (k_b < batch_size && k_io < inout_dim) {
		size_t idx = k_b * inout_dim + k_io;
		double exp_p = exp(input[idx]);
		double exp_n = exp(-input[idx]);
		output[idx] = (exp_p - exp_n) / (exp_p + exp_n);
	}
}

static __global__ void
kernel_tanh_fwd_default_ow_in_place_f64(size_t batch_size, size_t inout_dim,
					double *inout)
{
	size_t k_io = blockIdx.x * blockDim.x + threadIdx.x;
	size_t k_b = blockIdx.y * blockDim.y + threadIdx.y;

	if (k_b < batch_size && k_io < inout_dim) {
		size_t idx = k_b * inout_dim + k_io;
		double exp_p = exp(inout[idx]);
		double exp_n = exp(-inout[idx]);
		inout[idx] = (exp_p - exp_n) / (exp_p + exp_n);
	}
}

static __global__ void
kernel_tanh_fwd_default_accum_f64(size_t batch_size, size_t inout_dim,
				  const double *__restrict input,
				  double *__restrict output)
{
	size_t k_io = blockIdx.x * blockDim.x + threadIdx.x;
	size_t k_b = blockIdx.y * blockDim.y + threadIdx.y;

	if (k_b < batch_size && k_io < inout_dim) {
		size_t idx = k_b * inout_dim + k_io;
		double exp_p = exp(input[idx]);
		double exp_n = exp(-input[idx]);
		output[idx] += (exp_p - exp_n) / (exp_p + exp_n);
	}
}
#endif