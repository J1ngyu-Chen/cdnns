#include <cdnns_fused_linear.h>
#include <cuda_runtime.h>

static __global__ void kernel_cdnns_fused_linear_relu_ow_fp32(
	int batch_size, int in_dim, int out_dim, const float *RESTRICT input,
	const float *RESTRICT weights, const float *RESTRICT bias,
	float *RESTRICT output);

static __global__ void kernel_cdnns_fused_linear_no_bias_relu_ow_fp32(
	int batch_size, int in_dim, int out_dim, const float *RESTRICT input,
	const float *RESTRICT weights, float *RESTRICT output);

static __global__ void kernel_cdnns_fused_linear_relu_accum_fp32(
	int batch_size, int in_dim, int out_dim, const float *RESTRICT input,
	const float *RESTRICT weights, const float *RESTRICT bias,
	float *RESTRICT output);

static __global__ void kernel_cdnns_fused_linear_no_bias_relu_accum_fp32(
	int batch_size, int in_dim, int out_dim, const float *RESTRICT input,
	const float *RESTRICT weights, float *RESTRICT output);

static __global__ void kernel_cdnns_fused_linear_relu_ow_fp64(
	int batch_size, int in_dim, int out_dim, const double *RESTRICT input,
	const double *RESTRICT weights, const double *RESTRICT bias,
	double *RESTRICT output);

static __global__ void kernel_cdnns_fused_linear_no_bias_relu_ow_fp64(
	int batch_size, int in_dim, int out_dim, const double *RESTRICT input,
	const double *RESTRICT weights, double *RESTRICT output);

static __global__ void kernel_cdnns_fused_linear_relu_accum_fp64(
	int batch_size, int in_dim, int out_dim, const double *RESTRICT input,
	const double *RESTRICT weights, const double *RESTRICT bias,
	double *RESTRICT output);

static __global__ void kernel_cdnns_fused_linear_no_bias_relu_accum_fp64(
	int batch_size, int in_dim, int out_dim, const double *RESTRICT input,
	const double *RESTRICT weights, double *RESTRICT output);

extern "C" {

void cdnns_fused_linear_relu_ow_fp32(int batch_size, int in_dim, int out_dim,
				     const float *RESTRICT input,
				     const float *RESTRICT weights,
				     const float *RESTRICT bias,
				     float *RESTRICT output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((out_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_cdnns_fused_linear_relu_ow_fp32<<<grid_dim, block_dim>>>(
		batch_size, in_dim, out_dim, input, weights, bias, output);
}

void cdnns_fused_linear_no_bias_relu_ow_fp32(int batch_size, int in_dim,
					     int out_dim,
					     const float *RESTRICT input,
					     const float *RESTRICT weights,
					     float *RESTRICT output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((out_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_cdnns_fused_linear_no_bias_relu_ow_fp32<<<grid_dim, block_dim>>>(
		batch_size, in_dim, out_dim, input, weights, output);
}

void cdnns_fused_linear_relu_accum_fp32(int batch_size, int in_dim, int out_dim,
					const float *RESTRICT input,
					const float *RESTRICT weights,
					const float *RESTRICT bias,
					float *RESTRICT output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((out_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_cdnns_fused_linear_relu_accum_fp32<<<grid_dim, block_dim>>>(
		batch_size, in_dim, out_dim, input, weights, bias, output);
}

void cdnns_fused_linear_no_bias_relu_accum_fp32(int batch_size, int in_dim,
						int out_dim,
						const float *RESTRICT input,
						const float *RESTRICT weights,
						float *RESTRICT output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((out_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_cdnns_fused_linear_no_bias_relu_accum_fp32<<<grid_dim, block_dim>>>(
		batch_size, in_dim, out_dim, input, weights, output);
}

void cdnns_fused_linear_relu_ow_fp64(int batch_size, int in_dim, int out_dim,
				     const double *RESTRICT input,
				     const double *RESTRICT weights,
				     const double *RESTRICT bias,
				     double *RESTRICT output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((out_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_cdnns_fused_linear_relu_ow_fp64<<<grid_dim, block_dim>>>(
		batch_size, in_dim, out_dim, input, weights, bias, output);
}

void cdnns_fused_linear_no_bias_relu_ow_fp64(int batch_size, int in_dim,
					     int out_dim,
					     const double *RESTRICT input,
					     const double *RESTRICT weights,
					     double *RESTRICT output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((out_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_cdnns_fused_linear_no_bias_relu_ow_fp64<<<grid_dim, block_dim>>>(
		batch_size, in_dim, out_dim, input, weights, output);
}

void cdnns_fused_linear_relu_accum_fp64(int batch_size, int in_dim, int out_dim,
					const double *RESTRICT input,
					const double *RESTRICT weights,
					const double *RESTRICT bias,
					double *RESTRICT output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((out_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_cdnns_fused_linear_relu_accum_fp64<<<grid_dim, block_dim>>>(
		batch_size, in_dim, out_dim, input, weights, bias, output);
}

void cdnns_fused_linear_no_bias_relu_accum_fp64(int batch_size, int in_dim,
						int out_dim,
						const double *RESTRICT input,
						const double *RESTRICT weights,
						double *RESTRICT output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((out_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_cdnns_fused_linear_no_bias_relu_accum_fp64<<<grid_dim, block_dim>>>(
		batch_size, in_dim, out_dim, input, weights, output);
}
}

static __global__ void kernel_cdnns_fused_linear_relu_ow_fp32(
	int batch_size, int in_dim, int out_dim, const float *RESTRICT input,
	const float *RESTRICT weights, const float *RESTRICT bias,
	float *RESTRICT output)
{
	int o = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y * blockDim.y + threadIdx.y;

	if (b < batch_size && o < out_dim) {
		float sum = bias[o];
		for (int i = 0; i < in_dim; i++) {
			sum += input[b * in_dim + i] * weights[o * in_dim + i];
		}
		sum = (sum > 0.0f) * sum;
		output[b * out_dim + o] = sum;
	}
}

static __global__ void kernel_cdnns_fused_linear_no_bias_relu_ow_fp32(
	int batch_size, int in_dim, int out_dim, const float *RESTRICT input,
	const float *RESTRICT weights, float *RESTRICT output)
{
	int o = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y * blockDim.y + threadIdx.y;

	if (b < batch_size && o < out_dim) {
		float sum = 0.0f;
		for (int i = 0; i < in_dim; i++) {
			sum += input[b * in_dim + i] * weights[o * in_dim + i];
		}
		sum = (sum > 0.0f) * sum;
		output[b * out_dim + o] = sum;
	}
}

static __global__ void kernel_cdnns_fused_linear_relu_accum_fp32(
	int batch_size, int in_dim, int out_dim, const float *RESTRICT input,
	const float *RESTRICT weights, const float *RESTRICT bias,
	float *RESTRICT output)
{
	int o = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y * blockDim.y + threadIdx.y;

	if (b < batch_size && o < out_dim) {
		float sum = bias[o];
		for (int i = 0; i < in_dim; i++) {
			sum += input[b * in_dim + i] * weights[o * in_dim + i];
		}
		sum = (sum > 0.0f) * sum;
		output[b * out_dim + o] += sum;
	}
}

static __global__ void kernel_cdnns_fused_linear_no_bias_relu_accum_fp32(
	int batch_size, int in_dim, int out_dim, const float *RESTRICT input,
	const float *RESTRICT weights, float *RESTRICT output)
{
	int o = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y * blockDim.y + threadIdx.y;

	if (b < batch_size && o < out_dim) {
		float sum = 0.0f;
		for (int i = 0; i < in_dim; i++) {
			sum += input[b * in_dim + i] * weights[o * in_dim + i];
		}
		sum = (sum > 0.0f) * sum;
		output[b * out_dim + o] += sum;
	}
}

static __global__ void kernel_cdnns_fused_linear_relu_ow_fp64(
	int batch_size, int in_dim, int out_dim, const double *RESTRICT input,
	const double *RESTRICT weights, const double *RESTRICT bias,
	double *RESTRICT output)
{
	int o = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y * blockDim.y + threadIdx.y;

	if (b < batch_size && o < out_dim) {
		double sum = bias[o];
		for (int i = 0; i < in_dim; i++) {
			sum += input[b * in_dim + i] * weights[o * in_dim + i];
		}
		sum = (sum > 0.0) * sum;
		output[b * out_dim + o] = sum;
	}
}

static __global__ void kernel_cdnns_fused_linear_no_bias_relu_ow_fp64(
	int batch_size, int in_dim, int out_dim, const double *RESTRICT input,
	const double *RESTRICT weights, double *RESTRICT output)
{
	int o = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y * blockDim.y + threadIdx.y;

	if (b < batch_size && o < out_dim) {
		double sum = 0.0;
		for (int i = 0; i < in_dim; i++) {
			sum += input[b * in_dim + i] * weights[o * in_dim + i];
		}
		sum = (sum > 0.0) * sum;
		output[b * out_dim + o] = sum;
	}
}

static __global__ void kernel_cdnns_fused_linear_relu_accum_fp64(
	int batch_size, int in_dim, int out_dim, const double *RESTRICT input,
	const double *RESTRICT weights, const double *RESTRICT bias,
	double *RESTRICT output)
{
	int o = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y * blockDim.y + threadIdx.y;

	if (b < batch_size && o < out_dim) {
		double sum = bias[o];
		for (int i = 0; i < in_dim; i++) {
			sum += input[b * in_dim + i] * weights[o * in_dim + i];
		}
		sum = (sum > 0.0) * sum;
		output[b * out_dim + o] += sum;
	}
}

static __global__ void kernel_cdnns_fused_linear_no_bias_relu_accum_fp64(
	int batch_size, int in_dim, int out_dim, const double *RESTRICT input,
	const double *RESTRICT weights, double *RESTRICT output)
{
	int o = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y * blockDim.y + threadIdx.y;

	if (b < batch_size && o < out_dim) {
		double sum = 0.0;
		for (int i = 0; i < in_dim; i++) {
			sum += input[b * in_dim + i] * weights[o * in_dim + i];
		}
		sum = (sum > 0.0) * sum;
		output[b * out_dim + o] += sum;
	}
}