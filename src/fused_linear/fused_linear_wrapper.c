#include <cdnns_fused_linear.h>

void cdnns_fused_linear(enum cdnns_type type, enum cdnns_mode mode,
			enum cdnns_activation activation, int batch_size,
			int in_dim, int out_dim, const void *input,
			const void *weights, const void *bias, void *output)
{
	switch (type) {
	case CDNNS_TYPE_FP32:

		switch (mode) {
		case CDNNS_MODE_OVERWRITE:

			switch (activation) {
			case CDNNS_ACTIVATION_RELU:
				if (bias) {
					cdnns_fused_linear_relu_ow_fp32(
						batch_size, in_dim, out_dim,
						input, weights, bias, output);

				} else {
					cdnns_fused_linear_no_bias_relu_ow_fp32(
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
					cdnns_fused_linear_relu_accum_fp32(
						batch_size, in_dim, out_dim,
						input, weights, bias, output);

				} else {
					cdnns_fused_linear_no_bias_relu_accum_fp32(
						batch_size, in_dim, out_dim,
						input, weights, output);
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

/*
void cdnns_cdnns_fused_linear_backward(enum cdnns_type type, enum cdnns_mode mode,
			   enum cdnns_reduction reduction,
			   int batch_size, int in_dim, int out_dim,
			   const void *__restrict input,
			   void *__restrict input_grad,
			   const void *__restrict weights,
			   void *__restrict weights_grad,
			   void *__restrict bias_grad,
			   const void *__restrict output_grad)
{
#if defined USE_CBLAS_API
	// Compute the gradient of the input
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_size,
		    in_dim, out_dim, 1.0f, output_grad, out_dim, weights,
		    in_dim, 0.0f, input_grad, in_dim );

	// Branch on reduction to compute the gradients of the weights and bias
	if (reduction == CNeural_REDUCTION_MEAN) {
		// Compute the gradient of the weights
		cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, out_dim,
			    in_dim, batch_size, 1.0f / batch_size, output_grad,
			    out_dim, input, in_dim, 1.0f, weights_grad, in_dim);

		// Compute the gradient of the bias
		for (int b = 0; b < batch_size; b++) {
			cblas_saxpy(out_dim, 1.0f / batch_size, 
				    output_grad + b * out_dim, 1,
				    bias_grad, 1);
		}
	}
	else if (reduction == CNeural_REDUCTION_SUM) {
		// Compute the gradient of the weights
		cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, out_dim,
			    in_dim, batch_size, 1.0f, output_grad, out_dim,
			    input, in_dim, 1.0f, weights_grad, in_dim);

		// Compute the gradient of the bias
		for (int b = 0; b < batch_size; b++) {
			const float *og_b = output_grad + b * out_dim;
			cblas_saxpy(out_dim, 1.0f, og_b, 1, bias_grad, 1);
		}
	}
	else if (CNeural_REDUCTION_NONE) {
		// Compute the gradient of the weights
		CBLAS_TRANSPOSE TransA  = CblasTrans;
		CBLAS_TRANSPOSE TransB  = CblasNoTrans;
		cblasint M = out_dim;
		cblasint N = in_dim;
		cblasint K = 1;
		cblasint lda = out_dim;
		cblasint ldb = in_dim;;
		cblasint ldc = in_dim;
		float beta = 0.0f;
		float alpha = 1.0f;

		float **A_array = malloc(batch_size * sizeof(float*));
		if (A_array == NULL) {
			fprintf(stderr,
			"CNeural ERROR: Failed to allocate memory. "
			"Requested size: %zu bytes. Aborting.\n",
			batch_size * sizeof(float*));

			fflush(stderr);
			abort();
		}
		float **B_array = malloc(batch_size * sizeof(float*));
		if (B_array == NULL) {
			fprintf(stderr,
			"CNeural ERROR: Failed to allocate memory. "
			"Requested size: %zu bytes. Aborting.\n",
			batch_size * sizeof(float*));

			fflush(stderr);
			abort();
		}
		float **C_array = malloc(batch_size * sizeof(float*));
		if (C_array == NULL) {
			fprintf(stderr,
			"CNeural ERROR: Failed to allocate memory. "
			"Requested size: %zu bytes. Aborting.\n",
			batch_size * sizeof(float*));

			fflush(stderr);
			abort();
		}

		for (int b = 0; b < batch_size; b++) {
			A_array[b] = output_grad + b * out_dim;
			B_array[b] = input   + b * in_dim;   
			C_array[b] = weights  + b * out_dim * in_dim;
		}

		cblas_sgemm_batch(
			CblasRowMajor,
			&TransA, &TransB,
			&M, &N, &K,
			&alpha,
			A_array, &lda,
			B_array, &ldb,
			&beta,
			C_array, &ldc,
			1, &batch_size
		);
		free(A_array);
		free(B_array);
		free(C_array);

		// Compute the gradient of the bias
		cblas_saxpy(out_dim * batch_size, 1.0f, output_grad, 1,
			    bias_grad, 1);
	}
#else
	// Compute the gradient of the input
	float *in_g = input_grad;
	const float *out_g = output_grad;
	for (int b = 0; b < batch_size; b++) {
		const float *w = weights;
		for (int o = 0; o < out_dim; o++) {
			const float og = out_g[o];

			for (int i = 0; i < in_dim; i++)
				in_g[i] += og * w[i];
			w += in_dim;
		}

		in_g += in_dim;
		out_g += out_dim;
	}

	// Branch on reduction to compute the gradients of the weights and bias
	if (reduction == CDNNS_REDUCTION_MEAN) {
		// Compute the gradient of the weights and bias
		const float *in = input;
		const float *out_g = output_grad;
		for (int b = 0; b < batch_size; b++) {
			float *w_g = weights_grad;
			for (int o = 0; o < out_dim; o++) {
				for (int i = 0; i < in_dim; i++) {
					w_g[i] += (out_g[o] * in[i]) / batch_size;
				}
				bias_grad[o] += out_g[o] / batch_size;
				w_g += in_dim;
			}
			in += in_dim;
			out_g += out_dim;
		}
	}
	else if (reduction == CDNNS_REDUCTION_SUM) {
		// Compute the gradient of the weights and bias
		const float *in = input;
		const float *out_g = output_grad;
		for (int b = 0; b < batch_size; b++) {
			float *w_g = weights_grad;
			for (int o = 0; o < out_dim; o++) {
				for (int i = 0; i < in_dim; i++) {
					w_g[i] += out_g[o] * in[i];
				}
				bias_grad[o] += out_g[o];
				w_g += in_dim;
			}
			in += in_dim;
			out_g += out_dim;
		}
	}
	else if (CDNNS_REDUCTION_NONE) {
		// Compute the gradient of the weights and bias
		const float *in = input;
		const float *out_g = output_grad;
		float *w_g = weights_grad;
		float *b_g = bias_grad;
		for (int b = 0; b < batch_size; b++) {
			for (int o = 0; o < out_dim; o++) {
				for (int i = 0; i < in_dim; i++) {
					w_g[i] += out_g[o] * in[i];
				}
				b_g[o] += out_g[o];
				w_g += in_dim;
			}
			in += in_dim;
			b_g += out_dim;
			out_g += out_dim;
		}
	}
#endif
}

// NOT TESTED!
void cdnns_cdnns_fused_linear_backward_fp64(
	const double *__restrict input, double *__restrict input_grad,
	const double *__restrict weights, double *__restrict weights_grad,
	double *__restrict bias_grad,
	const double *__restrict output_grad,
	int batch_size,
	int in_dim, int out_dim,
	enum CNeural_reduction reduction)
{
#if defined USE_CBLAS_API
	// Compute the gradient of the input
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans, CblasNoTrans,
		batch_size, in_dim, out_dim,
		1.0,
		output_grad, out_dim,
		weights, in_dim,
		0.0,
		input_grad, in_dim
	);
	// Branch on reduction to compute the gradients of the weights and bias
	if (reduction == CNeural_REDUCTION_MEAN) {
		// Compute the gradient of the weights
		cblas_dgemm(
			CblasRowMajor,
			CblasTrans, CblasNoTrans,
			out_dim, in_dim, batch_size,
			1.0 / batch_size,
			output_grad, out_dim,
			input, in_dim,
			0.0,
			weights_grad, in_dim
		);
		// Compute the gradient of the bias
		for (int b = 0; b < batch_size; b++) {
			cblas_daxpy(
				out_dim,
				1.0 / batch_size,
				output_grad + b * out_dim, 1,
				bias_grad, 1
			);
		}
	}
	else if (reduction == CNeural_REDUCTION_SUM) {
		// Compute the gradient of the weights
		cblas_dgemm(
			CblasRowMajor,
			CblasTrans, CblasNoTrans,
			out_dim, in_dim, batch_size,
			1.0,
			output_grad, out_dim,
			input, in_dim,
			0.0,
			weights_grad, in_dim
		);
		// Compute the gradient of the bias
		for (int b = 0; b < batch_size; b++) {
			cblas_daxpy(
				out_dim,
				1.0,
				output_grad + b * out_dim, 1,
				bias_grad, 1
			);
		}
	}
	else if (CNeural_REDUCTION_NONE) {
		// Compute the gradient of the weights
		CBLAS_TRANSPOSE TransA = CblasTrans;
		CBLAS_TRANSPOSE TransB = CblasNoTrans;
		cblasint M = out_dim;
		cblasint N = in_dim;
		cblasint K = 1;
		cblasint lda = out_dim;
		cblasint ldb = in_dim;;
		cblasint ldc = in_dim;
		double alpha = 1.0;
		double beta = 0.0;

		double **A_array = malloc(batch_size * sizeof(double*));
		double **B_array = malloc(batch_size * sizeof(double*));
		double **C_array = malloc(batch_size * sizeof(double*));
		if (A_array == NULL || B_array == NULL || C_array == NULL) {
			int requested_num = 0;
			requested_num += (A_array == NULL);
			requested_num += (B_array == NULL);
			requested_num += (C_array == NULL);
	
			free(A_array);
			free(B_array);
			free(C_array);

			fprintf(stderr,
			"CNeural ERROR: Failed to allocate memory. "
			"Requested size: %zu bytes. Aborting.\n",
			batch_size * sizeof(double*));

			fflush(stderr);
			abort();
		}

		for (int b = 0; b < batch_size; b++) {
			A_array[b] = output_grad + b * out_dim;
			B_array[b] = input   + b * in_dim;
			C_array[b] = weights + b * out_dim * in_dim;
		}
		cblas_dgemm_batch(
			CblasRowMajor, &TransA, &TransB,
			&M, &N, &K,
			&alpha,
			A_array, &lda,
			B_array, &ldb,
			&beta,
			C_array, &ldc,
			1, &batch_size
		);

		free(A_array);
		free(B_array);
		free(C_array);

		// Compute the gradient of the bias
		cblas_daxpy(out_dim * batch_size,
			1.0,
			output_grad, 1,
			bias_grad, 1
		);
	}
#else
	// Compute the gradient of the input
	double *in_g = input_grad;
	const double *out_g = output_grad;
	for (int b = 0; b < batch_size; b++) {
		const double *w = weights;
		for (int o = 0; o < out_dim; o++) {
			const double og = out_g[o];

			for (int i = 0; i < in_dim; i++)
				in_g[i] += og * w[i];
			w += in_dim;
		}
		in_g += in_dim;
		out_g += out_dim;
	}

	// Branch on reduction to compute the gradients of the weights and bias
	if (reduction == CDNNS_REDUCTION_MEAN) {
		// Compute the gradient of the weights and bias
		const double *in = input;
		const double *out_g = output_grad;
		for (int b = 0; b < batch_size; b++) {
			double *w_g = weights_grad;
			for (int o = 0; o < out_dim; o++) {
				for (int i = 0; i < in_dim; i++) {
					w_g[i] += (out_g[o] * in[i]) / batch_size;
				}
				bias_grad[o] += out_g[o] / batch_size;
				w_g += in_dim;
			}
			in += in_dim;
			out_g += out_dim;
		}
	}
	else if (reduction == CDNNS_REDUCTION_SUM) {
		// Compute the gradient of the weights and bias
		const double *in = input;
		const double *out_g = output_grad;
		for (int b = 0; b < batch_size; b++) {
			double *w_g = weights_grad;
			for (int o = 0; o < out_dim; o++) {
				for (int i = 0; i < in_dim; i++) {
					w_g[i] += out_g[o] * in[i];
				}
				bias_grad[o] += out_g[o];
				w_g += in_dim;
			}
			in += in_dim;
			out_g += out_dim;
		}
	}
	else if (CDNNS_REDUCTION_NONE) {
		// Compute the gradient of the weights and bias
		const double *in = input;
		const double *out_g = output_grad;
		double *w_g = weights_grad;
		double *b_g = bias_grad;
		for (int b = 0; b < batch_size; b++) {
			for (int o = 0; o < out_dim; o++) {
				for (int i = 0; i < in_dim; i++) {
					w_g[i] += out_g[o] * in[i];
				}
				b_g[o] += out_g[o];
				w_g += in_dim;
			}
			in += in_dim;
			b_g += out_dim;
			out_g += out_dim;
		}
	}
#endif
}

*/
