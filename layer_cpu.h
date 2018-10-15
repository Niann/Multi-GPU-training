#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <random>

#include "cublas_v2.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void initialization(float* a, int size);
void printMatrix(const float* a, int r, int c);
void printGPUMatrix(const float* a, int r, int c);

class Layer_cpu {
protected:
	int l_prev, l_curr; // l_prev : neural number of previous layer, l : neural number of current layer
	float* W;  // weights, matrix of size (l_curr, l_prev), pointer on gpu
	float* dW; // W gradients
	float* b;  // bias, vector of size (l_curr,), pointer on gpu
	float* db; // b gradients
	float* dZ; // gradient of activation
			   // in forward pass, store dA/dZ
			   // in backward pass, dL/dZ = dL/dA * dA/dZ
	float* A_prev; // activation from previous layer, book-keeping for backward pass
public:
	Layer_cpu(int l1, int l2);
	virtual float* forward(float* X_in, int batch) = 0;
	virtual float* backward(float* dA, int batch) = 0;
	void SGDUpdate(float alpha);
	void freeMemory();

protected:
	float* WX_b(const float* W, const float* X, float* b, int m, int n, int k);
	void matrixMul(const float* A, const float* B, float* C, int m, int n, int k, bool transA, bool transB, float alpha, float beta);
	void reduceSum(float* A, float* b, int l, int batch, bool columnwise);
	void elementwiseMul(int numElements, float* A, const float* B, bool invB);
	void elementwiseAdd(int numElements, float* A, const float* B, float alpha);
	void elementwiseExp(float* A, int numElements);
	void broadcast(float* A, float* b, int r, int c, bool row);
};

class ReluLayer_cpu : public Layer_cpu {
public:
	ReluLayer_cpu(int l1, int l2);
	float* forward(float* X_in, int batch);
	float* backward(float* dA, int batch);
private:
	void relu(int numElements, float* Z, float* dZ);
};

class SoftmaxLayer_cpu : public Layer_cpu {
public:
	SoftmaxLayer_cpu(int l1, int l2);
	float* forward(float* X_in, int batch);
	float* backward(float* Y, int batch);
private:
	void softmax(int numElements, float* Z);
};