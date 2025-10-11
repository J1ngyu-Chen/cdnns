#ifndef CDNNS_H
#define CDNNS_H

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#include <cblas64.h>

#ifdef USE_ONEMKL
#include "mkl_cblas.h"
#define IMPL_CBLAS
typedef MKL_INT cblasint;
#endif

#ifdef USE_OPENBLAS
#include "cblas.h"
#define IMPL_CBLAS
typedef blasint cblasint;
#endif

enum cdnns_type {
	CDNNS_TYPE_FP32,
	CDNNS_TYPE_FP64,
};

enum cdnns_mode {
	CDNNS_MODE_OVERWRITE,
	CDNNS_MODE_OVERWRITE_IN_PLACE,
	CDNNS_MODE_ACCUMULATE
};

enum cdnns_option {
	CDNNS_OPTION_DEFAULT,
	CDNNS_OPTION_NO_BIAS,
	CDNNS_OPTION_FUSE_RELU
};

enum cdnns_reduction {
	CDNNS_REDUCTION_MEAN,
	CDNNS_REDUCTION_SUM,
	CDNNS_REDUCTION_NONE
};

void cdnns_elu_fwd(enum cdnns_type type, enum cdnns_mode mode,
		   enum cdnns_option option, size_t batch_size,
		   size_t inout_dim, const void *input, const void *alpha,
		   void *output);

void cdnns_leakyrelu_fwd(enum cdnns_type type, enum cdnns_mode mode,
			 enum cdnns_option option, size_t batch_size,
			 size_t inout_dim, const void *input, const void *alpha,
			 void *output);

void cdnns_linear_fwd(enum cdnns_type type, enum cdnns_mode mode,
		      enum cdnns_option option, size_t batch_size,
		      size_t in_dim, size_t out_dim, const void *input,
		      const void *weight, const void *bias, void *output);

void cdnns_relu_fwd(enum cdnns_type type, enum cdnns_mode mode,
		    enum cdnns_option option, size_t batch_size,
		    size_t inout_dim, const void *input, void *output);

void cdnns_sigmoid_fwd(enum cdnns_type type, enum cdnns_mode mode,
		       enum cdnns_option option, size_t batch_size,
		       size_t inout_dim, const void *input, void *output);

void cdnns_softmax_fwd(enum cdnns_type type, enum cdnns_mode mode,
		       enum cdnns_option option, size_t batch_size,
		       size_t inout_dim, const void *input, void *output);

void cdnns_tanh_fwd(enum cdnns_type type, enum cdnns_mode mode,
		    enum cdnns_option option, size_t batch_size,
		    size_t inout_dim, const void *input, void *output);

/*
void CNeural_leakyrelu_backward_f64(
double *input,
double *input_grad,
double *output_grad,
size_t batch_size, size_t inout_dim,
double alpha
);
*/

/*
void cdnns_linear_backward(enum cdnns_type type, enum cdnns_mode mode,
			   enum cdnns_reduction reduction,
			   size_t batch_size, size_t in_dim, size_t out_dim,
			   const void *restrict input,
			   void *restrict input_grad,
			   const void *restrict weight,
			   void *restrict weight_grad,
			   void *restrict bias_grad,
			   const void *restrict output_grad);
*/

/*
// NOT TESTED!
void CNeural_relu_backward(
	const float *input,
	float *input_grad,
	const float *output_grad,
	size_t batch_size, size_t inout_dim
);
*/

/*
void CNeural_sigmoid_backward_f32(
float *input,
float *input_grad,
float *output_grad,
size_t batch_size, size_t inout_dim
);
*/

/*
void CNeural_softmax_backward_f32(
float *input,
float *input_grad,
float *output_grad,
size_t batch_size, size_t inout_dim
);
*/

/*
void CNeural_tanh_backward_f64(
double *input,
double *input_grad,
double *output_grad,
size_t batch_size, size_t inout_dim
);
*/

/*
void CNeural_elu_backward_f64(
	double *input,
	double *input_grad,
	double *output_grad,
	size_t batch_size, size_t inout_dim,
	double alpha);
*/

#endif
