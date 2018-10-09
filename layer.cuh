#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <random>

#include "cublas_v2.h"
#ifndef LAYER_H
#define LAYER_H


#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void initialization(float* a, int size);

class Layer {
protected:
	int l_prev, l_curr; // l_prev : neural number of previous layer, l : neural number of current layer
	float* W;  // weights, matrix of size (l_curr, l_prev)
	float* dW; // W gradients
	float* b;  // bias, vector of size (l_curr,)
	float* db; // b gradients
	float* dZ; // gradient of activation
			   // in forward pass, store dA/dZ
			   // in backward pass, dL/dZ = dL/dA * dA/dZ
	const float* A_prev; // activation from previous layer, book-keeping for backward pass
public:
	Layer(int l1, int l2);
	float* forward(const float* X_in, int batch);
	float* backward(const float* dA, int batch);
	void gradientUpdate(float alpha);
	void freeMemory();

protected:
	float* WX_b(const float* W, const float* X, const float* b, int m, int n, int k);
	void matrixMul(const float* A, const float* B, float* C, int m, int n, int k, bool transA, bool transB, float alpha, float beta);
	void reduceSum(float* A, float* b, int l, int batch, bool columnwise);
	void elementwiseMul(int numElements, float* A, const float* B, bool invB);
	void elementwiseAdd(int numElements, float* A, const float* B, float alpha);
	void relu(int numElements, float* Z, float* dZ);
	void broadcast(float* c, const float* b, int l, int batch, bool row);
};

class SoftmaxLayer : public Layer {
public:
	SoftmaxLayer(int l1, int l2);
	float* forward(const float* X_in, int batch);
	float* backward(const float* Y, int batch);
private:
	void softmax(int numElements, float* Z);

};

#endif // !LAYER_H
