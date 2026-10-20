#ifndef CDNNS_H
#define CDNNS_H

#ifndef RESTRICT
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#define RESTRICT restrict
#elif defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict
#elif defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT
#endif
#endif

enum cdnns_type {
	CDNNS_TYPE_FP32,
	CDNNS_TYPE_FP64,
};

enum cdnns_mode { CDNNS_MODE_OVERWRITE = 0, CDNNS_MODE_ACCUMULATE = 1 };

enum cdnns_activation {
	CDNNS_ACTIVATION_ELU,
	CDNNS_ACTIVATION_LEAKYRELU,
	CDNNS_ACTIVATION_RELU,
	CDNNS_ACTIVATION_SIGMOID,
	CDNNS_ACTIVATION_SOFTMAX,
	CDNNS_ACTIVATION_TANH
};

struct cdnns_conv2d_option_struct {
	int padding_h;
	int padding_w;
	int dilation_h;
	int dilation_w;
	int stride_h;
	int stride_w;

	int groups;
};

void cdnns_conv2d(enum cdnns_type type, enum cdnns_mode mode, int batch_size,
		  int in_height, int in_width, int in_channels,
		  int out_channels, int kernel_height, int kernel_width,
		  struct cdnns_conv2d_option_struct *option,
		  const float *RESTRICT input, const float *RESTRICT kernel,
		  const float *RESTRICT bias, float *RESTRICT output);

void cdnns_elu(enum cdnns_type type, enum cdnns_mode mode, int batch_size,
	       int inout_dim, const void *input, const void *alpha,
	       void *output);

void cdnns_leakyrelu(enum cdnns_type type, enum cdnns_mode mode, int batch_size,
		     int inout_dim, const void *input, const void *alpha,
		     void *output);

void cdnns_linear(enum cdnns_type type, enum cdnns_mode mode, int batch_size,
		  int in_dim, int out_dim, const void *input,
		  const void *weights, const void *bias, void *output);

void cdnns_relu(enum cdnns_type type, enum cdnns_mode mode, int batch_size,
		int inout_dim, const void *input, void *output);

void cdnns_sigmoid(enum cdnns_type type, enum cdnns_mode mode, int batch_size,
		   int inout_dim, const void *input, void *output);

void cdnns_softmax(enum cdnns_type type, enum cdnns_mode mode, int batch_size,
		   int inout_dim, const void *input, void *output);

void cdnns_tanh(enum cdnns_type type, enum cdnns_mode mode, int batch_size,
		int inout_dim, const void *input, void *output);

/*
void CNeural_leakyrelu_backward_f64(
double *input,
double *input_grad,
double *output_grad,
int batch_size, int inout_dim,
double alpha
);
*/

/*
void cdnns_linear_backward(enum cdnns_type type, enum cdnns_mode mode,
			   enum cdnns_reduction reduction,
			   int batch_size, int in_dim, int out_dim,
			   const void *restrict input,
			   void *restrict input_grad,
			   const void *restrict weights,
			   void *restrict weights_grad,
			   void *restrict bias_grad,
			   const void *restrict output_grad);
*/

/*
// NOT TESTED!
void CNeural_relu_backward(
	const float *input,
	float *input_grad,
	const float *output_grad,
	int batch_size, int inout_dim
);
*/

/*
void CNeural_sigmoid_backward_f32(
float *input,
float *input_grad,
float *output_grad,
int batch_size, int inout_dim
);
*/

/*
void CNeural_softmax_backward_f32(
float *input,
float *input_grad,
float *output_grad,
int batch_size, int inout_dim
);
*/

/*
void CNeural_tanh_backward_f64(
double *input,
double *input_grad,
double *output_grad,
int batch_size, int inout_dim
);
*/

/*
void CNeural_elu_backward_f64(
	double *input,
	double *input_grad,
	double *output_grad,
	int batch_size, int inout_dim,
	double alpha);
*/

#endif
