#include <cdnns_sigmoid.h>

void cdnns_cdnns_sigmoid(enum cdnns_type type, enum cdnns_mode mode,
			 int batch_size, int inout_dim, const void *input,
			 void *output)
{
	switch (type) {
	case CDNNS_TYPE_FP32:

		switch (mode) {
		case CDNNS_MODE_OVERWRITE:

			cdnns_sigmoid_ow_fp32(batch_size, inout_dim, input,
					      output);
			break;

		case CDNNS_MODE_ACCUMULATE:

			cdnns_sigmoid_accum_fp32(batch_size, inout_dim, input,
						 output);
			break;
		}
		break;
	case CDNNS_TYPE_FP64:

		switch (mode) {
		case CDNNS_MODE_OVERWRITE:

			cdnns_sigmoid_ow_fp64(batch_size, inout_dim, input,
					      output);
			break;

		case CDNNS_MODE_ACCUMULATE:

			cdnns_sigmoid_accum_fp64(batch_size, inout_dim, input,
						 output);
			break;
		}
		break;
	}
}

/*
// NOT TESTED!
void CNeural_relu_backward_fp64(const double *input, double *input_grad,
			       const double *output_grad, int batch_size,
			       int inout_dim)
{
	// Compute the gradient of the input
	for (int b = 0; b < batch_size; b++) {
		for (int io = 0; io < inout_dim; io++) {
			input_grad[b  *inout_dim + io] += output_grad[b  *inout_dim + io]  *(input[b  *inout_dim + io] > 0.0);
		}
	}
}
*/