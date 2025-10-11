#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cdnns.h>

#include <cuda_runtime.h>

#define BATCH_SIZE 1

static float input[BATCH_SIZE * 28 * 28] = { 0 };

static float weight_1[100 * 28 * 28] = { 0 };
static float bias_1[100] = { 0 };

static float hidden[BATCH_SIZE * 100] = { 0 };

static float weight_2[10 * 100] = { 0 };
static float bias_2[10] = { 0 };

static float output[BATCH_SIZE * 10] = { 0 };

static void run()
{
	printf("Loading...\n");
	float *x_train = (float *)calloc(60000 * 28 * 28, sizeof(float));
	float *y_train = (float *)calloc(60000 * 10, sizeof(float));
	float *x_test = (float *)calloc(10000 * 28 * 28, sizeof(float));
	float *y_test = (float *)calloc(10000 * 10, sizeof(float));

	FILE *file_x_train = fopen("./test_params/x_train.txt", "r");
	FILE *file_y_train = fopen("./test_params/y_train.txt", "r");
	FILE *file_x_test = fopen("./test_params/x_test.txt", "r");
	FILE *file_y_test = fopen("./test_params/y_test.txt", "r");

	for (size_t i = 0; i < 60000 * 28 * 28; i++) {
		fscanf(file_x_train, "%f", &x_train[i]);
	}

	for (size_t i = 0; i < 60000 * 10; i++) {
		fscanf(file_y_train, "%f", &y_train[i]);
	}

	for (size_t i = 0; i < 10000 * 28 * 28; i++) {
		fscanf(file_x_test, "%f", &x_test[i]);
	}

	for (size_t i = 0; i < 10000 * 10; i++) {
		fscanf(file_y_test, "%f", &y_test[i]);
	}

	fclose(file_x_train);
	fclose(file_y_train);
	fclose(file_x_test);
	fclose(file_y_test);

	FILE *file_weight_1 = fopen("./test_params/weight_1.txt", "r");
	FILE *file_bias_1 = fopen("./test_params/bias_1.txt", "r");
	FILE *file_weight_2 = fopen("./test_params/weight_2.txt", "r");
	FILE *file_bias_2 = fopen("./test_params/bias_2.txt", "r");

	for (size_t i = 0; i < 100 * 28 * 28; i++) {
		fscanf(file_weight_1, "%f", &weight_1[i]);
	}

	for (size_t i = 0; i < 100; i++) {
		fscanf(file_bias_1, "%f", &bias_1[i]);
	}

	for (size_t i = 0; i < 10 * 100; i++) {
		fscanf(file_weight_2, "%f", &weight_2[i]);
	}

	for (size_t i = 0; i < 10; i++) {
		fscanf(file_bias_2, "%f", &bias_2[i]);
	}

	fclose(file_weight_1);
	fclose(file_bias_1);
	fclose(file_weight_2);
	fclose(file_bias_2);

	float *d_input = NULL;
	float *d_weight_1 = NULL;
	float *d_bias_1 = NULL;
	float *d_hidden = NULL;
	float *d_weight_2 = NULL;
	float *d_bias_2 = NULL;
	float *d_output = NULL;

	cudaMalloc(&d_input, 28 * 28 * sizeof(float));
	cudaMalloc(&d_weight_1, 100 * 28 * 28 * sizeof(float));
	cudaMalloc(&d_bias_1, 100 * sizeof(float));
	cudaMalloc(&d_hidden, 100 * sizeof(float));
	cudaMalloc(&d_weight_2, 10 * 100 * sizeof(float));
	cudaMalloc(&d_bias_2, 10 * sizeof(float));
	cudaMalloc(&d_output, 10 * sizeof(float));

	printf("Loading completed\n");

	while (1) {
		int which_one;
		printf("input a number to run. The number must be from 0 to 9999\n");
		scanf("%d", &which_one);
		getchar();
		if (which_one < 0 || which_one > 9999) {
			continue;
		}

		memcpy(input, x_test + which_one * 28 * 28,
		       28 * 28 * sizeof(float));

		cudaMemcpy(d_weight_1, weight_1, 100 * 28 * 28 * sizeof(float),
			   cudaMemcpyHostToDevice);
		cudaMemcpy(d_bias_1, bias_1, 100 * sizeof(float),
			   cudaMemcpyHostToDevice);
		cudaMemcpy(d_hidden, hidden, 100 * sizeof(float),
			   cudaMemcpyHostToDevice);
		cudaMemcpy(d_weight_2, weight_2, 10 * 100 * sizeof(float),
			   cudaMemcpyHostToDevice);

		cudaMemcpy(d_input, input, 28 * 28 * sizeof(float),
			   cudaMemcpyHostToDevice);

		cdnns_linear_fwd(CDNNS_TYPE_FP32, CDNNS_MODE_OVERWRITE,
				 CDNNS_OPTION_FUSE_RELU, 1, 28 * 28, 100,
				 d_input, d_weight_1, d_bias_1, d_hidden);

		cdnns_linear_fwd(CDNNS_TYPE_FP32, CDNNS_MODE_OVERWRITE,
				 CDNNS_OPTION_DEFAULT, 1, 100, 10, d_hidden,
				 d_weight_2, d_bias_2, d_output);

		cdnns_softmax_fwd(CDNNS_TYPE_FP32,
				  CDNNS_MODE_OVERWRITE_IN_PLACE,
				  CDNNS_OPTION_DEFAULT, 1, 10, d_output,
				  d_output);

		cudaMemcpy(output, d_output, 10 * sizeof(float),
			   cudaMemcpyDeviceToHost);

		for (size_t i = 0; i < 28; i++) {
			for (size_t j = 0; j < 28; j++) {
				printf("%.0f ", input[i * 28 + j]);
			}
			printf("\n");
		}
		for (size_t i = 0; i < 10; i++) {
			printf("%zu: %f\n", i, output[i]);
		}

		while (1) {
			char yes_or_no;
			printf("Continue or not?(y/n)\n");
			scanf("%c", &yes_or_no);
			if (yes_or_no == 'y') {
				break;
			} else if (yes_or_no == 'n') {
				printf("Finish\n");
				return;
			}
		}
	}

	cudaFree(d_input);
	cudaFree(d_weight_1);
	cudaFree(d_bias_1);
	cudaFree(d_hidden);
	cudaFree(d_weight_2);
	cudaFree(d_bias_2);
	cudaFree(d_output);
}

int main()
{
	run();

	return 0;
}
