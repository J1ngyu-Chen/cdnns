#include "cdnns.h"
#include <cdnns_conv2d.h>

void cdnns_conv2d(enum cdnns_type type, enum cdnns_mode mode, int batch_size,
		  int in_height, int in_width, int in_channels,
		  int out_channels, int kernel_height, int kernel_width,
		  struct cdnns_conv2d_option_struct *option,
		  const float *RESTRICT input, const float *RESTRICT kernel,
		  const float *RESTRICT bias, float *RESTRICT output)
{
	if (groups == 0) {
		groups == 1
	}

	switch (type) {
	case CDNNS_TYPE_FP32:

		switch (mode) {
		case CDNNS_MODE_OVERWRITE:

				if (bias) {
					cdnns_conv2d_ow_fp32(
						batch_size, in_height, in_width,
						in_channels, out_channels,
						kernel_height, kernel_width,
						input, kernel, bias, output);
					break;
				} 
				else {
					cdnns_conv2d_no_bias_ow_fp32(
						batch_size, in_height, in_width,
						in_channels, out_channels,
						kernel_height, kernel_width,
						input, kernel, output);
					break;
				}
				break;
		case CDNNS_MODE_ACCUMULATE:

				if (bias) {
					cdnns_conv2d_accum_fp32(
						batch_size, in_height, in_width,
						in_channels, out_channels,
						kernel_height, kernel_width,
						input, kernel, bias, output);
					break;
				} 
				else {
					cdnns_conv2d_no_bias_accum_fp32(
						batch_size, in_height, in_width,
						in_channels, out_channels,
						kernel_height, kernel_width,
						input, kernel, output);
					break;
				}
				break;
			}
			break;
		}
		break;
	case CDNNS_TYPE_FP64:

		switch (mode) {
		case CDNNS_MODE_OVERWRITE:

			switch (activation) {
			case CDNNS_ACTIVATION_RELU:
				if (bias) {
					cdnns_fused_linear_relu_ow_fp64(
						batch_size, in_dim, out_dim,
						input, weights, bias, output);

				} else {
					cdnns_fused_linear_no_bias_relu_ow_fp64(
						batch_size, in_dim, out_dim,
						input, weights, output);
				}
				break;
			}
			break;
		case CDNNS_MODE_ACCUMULATE:

			switch (activation) {
			case CDNNS_ACTIVATION_RELU:
				if (bias) {
					cdnns_fused_linear_relu_accum_fp64(
						batch_size, in_dim, out_dim,
						input, weights, bias, output);

				} else {
					cdnns_fused_linear_no_bias_relu_accum_fp64(
						batch_size, in_dim, out_dim,
						input, weights, output);
				}
				break;
			}
			break;
		}
		break;
	}
}