#include <cdnns.h>

void cdnns_conv2d_ow_fp32(int batch_size, int in_height, int in_width, int in_channels,
		       int out_channels, int kernel_height, int kernel_width,
		       const float *RESTRICT input,
		       const float *RESTRICT kernel, const float *RESTRICT bias,
		       float *RESTRICT output)
{
        int out_height = (in_height - kernel_height) + 1;
        int out_width = (in_width - kernel_width) + 1;

	for (int b = 0; b < batch_size; b++)
        for (int cout = 0; cout < out_channels; cout++) {
                for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                        float sum = 0;
                        for (int cin = 0; cin < in_channels; cin++) {
                        for (int kh = 0; kh < kernel_height; kh++) {
                        for (int kw = 0; kw < kernel_width; kw++) {
                                sum += kernel[cout * in_channels * kernel_height * kernel_width + cin * kernel_height * kernel_width + kh * kernel_width + kw] * 
                                input[b * in_channels * in_height * in_width + cin * in_height * in_width + (oh + kh) * in_width + (ow + kw)];
                        }} }
                        sum += bias[cout];
                        output[b * out_channels * out_height * out_width + cout * out_height * out_width + oh * out_width + ow] = sum;
                }}
        }
}

void cdnns_conv2d_no_bias_ow_fp32(int batch_size, int in_height, int in_width, int in_channels,
		       int out_channels, int kernel_height, int kernel_width,
		       const float *RESTRICT input,
		       const float *RESTRICT kernel,
		       float *RESTRICT output)
{
        int out_height = (in_height - kernel_height) + 1;
        int out_width = (in_width - kernel_width) + 1;

	for (int b = 0; b < batch_size; b++)
        for (int cout = 0; cout < out_channels; cout++) {
                for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                        float sum = 0;
                        for (int cin = 0; cin < in_channels; cin++) {
                        for (int kh = 0; kh < kernel_height; kh++) {
                        for (int kw = 0; kw < kernel_width; kw++) {
                                sum += kernel[cout * in_channels * kernel_height * kernel_width + cin * kernel_height * kernel_width + kh * kernel_width + kw] * 
                                input[b * in_channels * in_height * in_width + cin * in_height * in_width + (oh + kh) * in_width + (ow + kw)];
                        }} }
                        output[b * out_channels * out_height * out_width + cout * out_height * out_width + oh * out_width + ow] = sum;
                }}
        }
}

void cdnns_conv2d_accum_fp32(int batch_size, int in_height, int in_width, int in_channels,
		       int out_channels, int kernel_height, int kernel_width,
		       const float *RESTRICT input,
		       const float *RESTRICT kernel, const float *RESTRICT bias,
		       float *RESTRICT output)
{
        int out_height = (in_height - kernel_height) + 1;
        int out_width = (in_width - kernel_width) + 1;

	for (int b = 0; b < batch_size; b++)
        for (int cout = 0; cout < out_channels; cout++) {
                for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                        float sum = 0;
                        for (int cin = 0; cin < in_channels; cin++) {
                        for (int kh = 0; kh < kernel_height; kh++) {
                        for (int kw = 0; kw < kernel_width; kw++) {
                                sum += kernel[cout * in_channels * kernel_height * kernel_width + cin * kernel_height * kernel_width + kh * kernel_width + kw] * 
                                input[b * in_channels * in_height * in_width + cin * in_height * in_width + (oh + kh) * in_width + (ow + kw)];
                        }} }
                        sum += bias[cout];
                        output[b * out_channels * out_height * out_width + cout * out_height * out_width + oh * out_width + ow] += sum;
                }}
        }
}

void cdnns_conv2d_no_bias_accum_fp32(int batch_size, int in_height, int in_width, int in_channels,
		       int out_channels, int kernel_height, int kernel_width,
		       const float *RESTRICT input,
		       const float *RESTRICT kernel,
		       float *RESTRICT output)
{
        int out_height = (in_height - kernel_height) + 1;
        int out_width = (in_width - kernel_width) + 1;

	for (int b = 0; b < batch_size; b++)
        for (int cout = 0; cout < out_channels; cout++) {
                for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                        float sum = 0;
                        for (int cin = 0; cin < in_channels; cin++) {
                        for (int kh = 0; kh < kernel_height; kh++) {
                        for (int kw = 0; kw < kernel_width; kw++) {
                                sum += kernel[cout * in_channels * kernel_height * kernel_width + cin * kernel_height * kernel_width + kh * kernel_width + kw] * 
                                input[b * in_channels * in_height * in_width + cin * in_height * in_width + (oh + kh) * in_width + (ow + kw)];
                        }} }
                        output[b * out_channels * out_height * out_width + cout * out_height * out_width + oh * out_width + ow] += sum;
                }}
        }
}

void cdnns_conv2d__fp32(int batch_size, int groups, int in_height, int in_width, int in_channels,
		       int out_channels, int kernel_height, int kernel_width, int dilation, int stride_h, int stride_w,
		       const float *RESTRICT input,
		       const float *RESTRICT kernel, const float *RESTRICT bias,
		       float *RESTRICT output)
{
        int out_height = (in_height - kernel_height) / stride_h + 1;
        int out_width = (in_width - kernel_width) / stride_h + 1;
	for (int b = 0; b < batch_size; b++)
        for (int g = 0; g < groups; g++)
        for (int cout = g * (out_channels / groups); cout < (g + 1) * (out_channels / groups); cout++) {
                for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                        float sum = 0;
                        for (int cin = g * (in_channels / groups); cin < (g + 1) * (in_channels / groups); cin++) {
                        for (int kh = 0; kh < kernel_height; kh++) {
                        for (int kw = 0; kw < kernel_width; kw++) {
                                sum += kernel[cout * in_channels * kernel_height * kernel_width + cin * kernel_height * kernel_width + kh * kernel_width + kw] * 
                                input[b * in_channels * in_height * in_width + cin * in_height * in_width + (oh * stride_h + kh) * in_width + (ow * stride_w + kw)];
                        }} }
                        sum += bias[cout];
                        output[b * out_channels * out_height * out_width + cout * out_height * out_width + oh * out_width + ow] = sum;
                }}
    
        }
}

void cdnns_conv2d_fp64(int batch_size, int in_dim, int out_dim,
		       const double *RESTRICT input,
		       const double *RESTRICT weight,
		       const double *RESTRICT bias, double *RESTRICT output)
{
	for (int b = 0; b < batch_size; b++) {
		for (int o = 0; o < out_dim; o++) {
			double sum = bias[o];
			for (int i = 0; i < in_dim; i++) {
				sum += input[b * in_dim + i] *
				       weight[o * in_dim + i];
			}
			output[b * out_dim + o] = sum;
		}
	}
}