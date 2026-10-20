#ifndef CDNNS_CONV2D_H
#define CDNNS_CONV2D_H
#include <cdnns.h>

void cdnns_conv2d_ow_fp32(int batch_size, int in_height, int in_width,
			  int in_channels, int out_channels, int kernel_height,
			  int kernel_width, const float *RESTRICT input,
			  const float *RESTRICT kernel,
			  const float *RESTRICT bias, float *RESTRICT output);

void cdnns_conv2d_no_bias_ow_fp32(int batch_size, int in_height, int in_width,
				  int in_channels, int out_channels,
				  int kernel_height, int kernel_width,
				  const float *RESTRICT input,
				  const float *RESTRICT kernel,
				  float *RESTRICT output);

void cdnns_conv2d_accum_fp32(int batch_size, int in_height, int in_width,
			     int in_channels, int out_channels,
			     int kernel_height, int kernel_width,
			     const float *RESTRICT input,
			     const float *RESTRICT kernel,
			     const float *RESTRICT bias,
			     float *RESTRICT output);

void cdnns_conv2d_no_bias_accum_fp32(int batch_size, int in_height,
				     int in_width, int in_channels,
				     int out_channels, int kernel_height,
				     int kernel_width,
				     const float *RESTRICT input,
				     const float *RESTRICT kernel,
				     float *RESTRICT output);

#endif