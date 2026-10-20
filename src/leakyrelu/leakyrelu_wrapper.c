#include <cdnns_leakyrelu.h>

void cdnns_leakyrelu(enum cdnns_type type, enum cdnns_mode mode, int batch_size,
		     int inout_dim, const void *input, const void *alpha,
		     void *output)
{
	switch (type) {
	case CDNNS_TYPE_FP32:

		switch (mode) {
		case CDNNS_MODE_OVERWRITE:

			cdnns_leakyrelu_ow_fp32(batch_size, inout_dim, input,
						*(float*)alpha, output);
			break;

		case CDNNS_MODE_ACCUMULATE:

			cdnns_leakyrelu_accum_fp32(batch_size, inout_dim, input,
						   *(float*)alpha, output);
			break;
		}
		break;
	case CDNNS_TYPE_FP64:

		switch (mode) {
		case CDNNS_MODE_OVERWRITE:

			cdnns_leakyrelu_ow_fp64(batch_size, inout_dim, input,
						*(double*)alpha, output);
			break;

		case CDNNS_MODE_ACCUMULATE:

			cdnns_leakyrelu_accum_fp64(batch_size, inout_dim, input,
						   *(float*)alpha, output);
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
                        input_grad[b  *inout_dim + io] += output_grad[b
*inout_dim + io]  *(input[b  *inout_dim + io] > 0.0);
                }
        }
}
*/