#include "layer_cpu.h"

using namespace std;
#define IDX2C(i, j, ld) ((( j )*( ld ))+( i )) // ld - leading dimension
#define IDX2C_T(i, j, ld) ((( i )*( ld ))+( j )) // trans

std::default_random_engine generator;
std::normal_distribution<float> distribution(0.0f, 0.005f);

void initialization(float* a, int size) {
	for (int i = 0; i < size; i++) {
		a[i] = distribution(generator);
	}
}

void printMatrix(const float* a, int r, int c) {
	// print matrix row by row, debugging purpose
	// r - number of rows
	// c - number of columns
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			printf(" %6.3f", a[IDX2C(i, j, r)]);
		}
		printf("\n");
	}
	printf("\n");
}

void reluHelper(float* Z, float* dZ, int numElements) {
	// perform relu activation and calculate gradients simultaneously

	for (int i = 0; i < numElements;i++) {
		if (Z[i] < 0) {
			Z[i] = 0;
			dZ[i] = 0;
		}
		else {
			dZ[i] = 1;
		}
	}
}

void broadcastHelper(float* A, const float* b, int r, int c, bool row) {
	// broadcast b to A by row/column
	for (int i = 0; i < r * c; i++) {
		if (row) {
			A[i] = b[i % r];
		}
		else {
			A[i] = b[i / r];
		}
	}
}

void elementMulHelper(float* A, const float* B, int numElements, bool invB) {
	// perform relu activation and calculate gradients simultaneously
	for (int i = 0; i < numElements; i++) {
		if (invB) {
			A[i] /= B[i];
		}
		else {
			A[i] *= B[i];
		}
	}
}

void expHelper(float* A, int numElements) {
	// perform element-wise exp
	for (int i = 0; i < numElements; i++) {
		A[i] = exp(A[i]);
	}
}

void elementAddHelper(float* A, const float* B, float alpha, int numElements) {
	// perform element-wise addition
	for (int i = 0; i < numElements; i++) {
		A[i] += alpha * B[i];
	}
}

void matrixMulHelper(const float* A, const float* B, float* C, int m, int n, int k, float alpha) {
	// perform relu activation and calculate gradients simultaneously
	int i, j, k_;
	#pragma omp parallel shared(A,B,C) private(i,j,k)
	{
	#pragma omp for schedule(dynamic)
		for (i = 0; i < m; i++)
		{
			for (j = 0; j < n; j++)
			{
				C[IDX2C(i,j,m)] = 0;
				for (k_ = 0; k_ < k; k_++)
				{
					C[IDX2C(i, j, m)] += alpha * A[IDX2C(i, k_, m)] * B[IDX2C(k_, j, k)];
				}
			}
		}
	}
}

void transpose(float* A, const float *B, int m, int n) {
	#pragma omp for
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			A[IDX2C_T(i, j, n)] = B[IDX2C(i, j, m)];
		}
	}
}

void copyMat(float* A, const float *B, int numElements) {
	for (int i = 0; i < numElements; i++) {
		A[i] += B[i];
	}
}


Layer_cpu::Layer_cpu(int l1, int l2) {
	l_prev = l1;
	l_curr = l2;

	// allocate memory
	float* h_W = (float *)malloc(l1 * l2 * sizeof(float));
	float* h_b = (float *)malloc(l2 * sizeof(float));

	float* W = (float *)malloc(l1 * l2 * sizeof(float));
	float* dW = (float *)malloc(l1 * l2 * sizeof(float));
	float* b = (float *)malloc(l2 * sizeof(float));
	float* db = (float *)malloc(l2 * sizeof(float));

	// initialize W and b
	initialization(W, l1 * l2);
	initialization(b, l2);
}

ReluLayer_cpu::ReluLayer_cpu(int l1, int l2) : Layer_cpu(l1, l2) {}

SoftmaxLayer_cpu::SoftmaxLayer_cpu(int l1, int l2) : Layer_cpu(l1, l2) {}

float* ReluLayer_cpu::forward(float* X_in, int batch) {
	// X_in input matrix, size (l_prev, batch_size), each column is a data point
	A_prev = X_in; // save activation from previous layer for backprop
	float* X_out = WX_b(W, X_in, b, l_curr, batch, l_prev); // X_out = Z = W @ X + b

	// allocate memory for dZ (gradient of activation) and perform activation
	int numElements = l_curr * batch;
	dZ = (float *)malloc(numElements * sizeof(float));
	relu(numElements, X_out, dZ);

	return X_out; // X_out = A = relu(Z)
}

float* SoftmaxLayer_cpu::forward(float* X_in, int batch) {
	// X_in input matrix, size (l_prev, batch_size), each column is a data point
	A_prev = X_in; // save activation from previous layer for backprop
	float* X_out = WX_b(W, X_in, b, l_curr, batch, l_prev); // X_out = Z = W @ X + b

	// allocate memory for dZ and perform activation
	int numElements = l_curr * batch;
	softmax(numElements, X_out);
	dZ = X_out; // store for backprop

	return X_out; // X_out = softmax(Z)
}

float* ReluLayer_cpu::backward(float* dA, int batch) {
	// dA input matrix, size (l_curr, batch_size), each column is gradient of a datapoint of current layer
	// dA_prev output matrix
	float* dA_prev;
	int numElements = l_prev * batch;
	dA_prev = (float *)malloc(numElements * sizeof(float));

	// calculate dZ, dW, db, dA_prev
	elementwiseMul(l_curr * batch, dZ, dA, false);                                        // dZ = dL/dA * dA/dZ
	matrixMul(dZ, A_prev, dW, l_curr, l_prev, batch, false, true, 1 / (float)batch, 0.f); // dW = dL/dZ * dZ/dW = 1/m * dZ @ A_prev.T
	reduceSum(dZ, db, l_curr, batch, false);                                              // db = dL/dZ * dZ/db = 1/m * sum(dZ, axis=1)
	matrixMul(W, dZ, dA_prev, l_prev, batch, l_curr, true, false, 1.0f, 0.f);             // dA_prev = dL/dZ * dZ/dA_prev = W.T @ dZ

	free(dA);
	free(dZ);
	return dA_prev;
}

float* SoftmaxLayer_cpu::backward(float* Y, int batch) {
	// y - one-hot matrix of size (l_curr, batch)
	// each column is a one_hot label for corresponding forward data
	// dA_prev out_put matrix - gradient of activation from previous layer
	float* dA_prev;
	int numElements = l_prev * batch;
	dA_prev = (float *)malloc(numElements * sizeof(float));

	// calculate dZ, dW, db, dA_prev
	elementwiseAdd(l_curr * batch, dZ, Y, -1.0f);                                         // dZ = P - Y
	matrixMul(dZ, A_prev, dW, l_curr, l_prev, batch, false, true, 1 / (float)batch, 0.f); // dW = 1/m * dZ @ A_prev.T
	reduceSum(dZ, db, l_curr, batch, false);                                              // db = 1/m * sum(dZ, axis=1)
	matrixMul(W, dZ, dA_prev, l_prev, batch, l_curr, true, false, 1.0f, 0.f);             // dA = W.T @ dZ

	free(dZ);
	return dA_prev;
}

void Layer_cpu::SGDUpdate(float alpha) {
	// perform parameter update w.r.t to gradient direction with learning rate alpha
	elementwiseAdd(l_curr * l_prev, W, dW, -alpha);
	elementwiseAdd(l_curr, b, db, -alpha);
}

void Layer_cpu::freeMemory() {
	// release memory
	free(W);
	free(dW);
	free(b);
	free(db);
}


// helper functions
float* Layer_cpu::WX_b(const float* W, const float* X, float* b, int m, int n, int k) {
	// perform W @ X + b in a batch
	// m - l_curr, n - batch_size, k - l_prev
	// W is matrix of size (m, k)
	// X is matrix of size (k, n)
	// b is vector of size (m,)
	// c is matrix of size (m, n)
	float * c; // c - c on the device
	c = (float *)malloc(m*n * sizeof(*c));

	broadcast(c, b, m, n, true); // broadcast b

	matrixMul(W, X, c, m, n, k, false, false, 1.0f, 1.0f);

	return c;
}

void Layer_cpu::matrixMul(const float* A, const float* B, float* C, int m, int n, int k, bool transA, bool transB, float alpha, float beta) {
	// C = op(A) @ op(B) + beta * C
	// op(A) is matrix of size (m, k)
	// op(B) is matrix of size (k, n)
	//   C   is matrix of size (m, n)
	// modifies content of C in-place

		
	// matrix - matrix multiplication : d_c = alpha * op(d_a) @ op(d_b) + beta * d_c
	// op(d_a) - m x k matrix , op(d_b) - k x n matrix , d_c - m x n matrix
	// alpha, beta read from argument
	
	float *tA = (float *)malloc(m*k * sizeof(float));
	float *tB = (float *)malloc(k*n * sizeof(float));

	if (transA) {
		transpose(tA, A, k, m);
	}
	else {
		copyMat(tA, A, m*k);
	}

	if (transB) {
		transpose(tB, B, n, k);
	}
	else {
		copyMat(tB, B, k*n);
	}
	
	matrixMulHelper(tA, tB, C, m, n, k, alpha);
	free(tA);
	free(tB);

}

void Layer_cpu::reduceSum(float* A, float* b, int l, int batch, bool columnwise) {
	// reduce sum row-wise or column-wise
	// store results in b
	// get b by matrix - vector multiplication

	// create a vector of same size as b filled with 1
	float* x;
	float* d_x;
	float alpha;
	if (columnwise) {
		x = (float *)malloc(l * sizeof(float));
		alpha = 1.0f;
		for (int i = 0; i < l; i++)
			x[i] = 1;

		d_x = (float *)malloc(l * sizeof(float));

		matrixMul(A, d_x, b, batch, 1, l, true, false, alpha, 0.f);
	}
	else {
		x = (float *)malloc(batch * sizeof(float));
		alpha = 1 / (float)batch;
		for (int i = 0; i < batch; i++)
			x[i] = 1;

		d_x = (float *)malloc(batch * sizeof(float));

		matrixMul(A, d_x, b, l, 1, batch, false, false, alpha, 0.f);
	}
	free(x);
}

void Layer_cpu::elementwiseMul(int numElements, float* A, const float* B, bool invB) {
	// element-wise matrix multiplication, store results in A
	// A = A * B
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	elementMulHelper(A, B, numElements, invB);
}

void Layer_cpu::elementwiseAdd(int numElements, float* A, const float* B, float alpha) {
	// element-wise matrix/vector addtion
	// A = A + alpha * B
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	elementAddHelper(A, B, alpha, numElements);
}

void Layer_cpu::elementwiseExp(float* A, int numElements) {
	// element-wise exponential
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	expHelper(A, numElements);
}

void Layer_cpu::broadcast(float* A, float* b, int r, int c, bool row) {
	// broadcast b to A by row/column
	int threadsPerBlock = 256;
	int blocksPerGrid = (r * c + threadsPerBlock - 1) / threadsPerBlock;
	broadcastHelper(A, b, r, c, row);
}

void ReluLayer_cpu::relu(int numElements, float* Z, float* dZ) {
	// perform relu activation and calculate gradients simultaneously
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	reluHelper(Z, dZ, numElements);
}

void SoftmaxLayer_cpu::softmax(int numElements, float* Z) {
	// softmax operation over each coloum of Z
	// store gradients in dZ
	// 1st x = exp(x) for each element x in Z
	// 2nd p = sum(x) for each column in Z
	// 3rd x = x/p for each element each column in Z
	int batch = numElements / l_curr;

	// 1st x = exp(x) for each element x in Z
	elementwiseExp(Z, numElements);

	// 2nd p = sum(x) for each column in Z
	float* p; // vector length batch_size
	p = (float *)malloc(batch * sizeof(float));
	reduceSum(Z, p, l_curr, batch, true);

	// 3rd x = x/p for each element each column in Z
	float * P;
	P = (float *)malloc(numElements * sizeof(float));
	broadcast(P, p, l_curr, batch, false);
	elementwiseMul(numElements, Z, P, true);

	free(p);
	free(P);
}

/*
int main() {
	int batch = 5;
	int feature = 10;
	int l1 = 6;
	int l2 = 5;

	float* X;
	float* Y;
	X = (float *)malloc(feature * batch * sizeof(float));
	initialization(X, feature * batch);
	Y = (float *)malloc(l2 * batch * sizeof(float));
	initialization(Y, l2 * batch);
	printf("foward pass\n");
	printf("input matrix:\n");
	printMatrix(X, feature, batch);
	float* d_X;
	float* d_Y;
	cudaMalloc((void**)& d_X, feature * batch * sizeof(float));
	cudaMalloc((void**)& d_Y, l2 * batch * sizeof(float));
	cudaMemcpy(d_X, X, feature * batch * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Y, Y, l2 * batch * sizeof(float), cudaMemcpyHostToDevice);

	Layer* l = new ReluLayer(feature, l1);
	Layer* s = new SoftmaxLayer(l1, l2);
	float* X1 = l->forward(d_X, batch);
	printf("output matrix:\n");
	printGPUMatrix(X1, l1, batch);

	float* X2 = s->forward(X1, batch);
	printf("output matrix:\n");
	printGPUMatrix(X2, l2, batch);
	
	printf("\nbackward pass\n");
	printf("input matrix:\n");
	printGPUMatrix(d_Y, l2, batch);

	float* dA1 = s->backward(d_Y, batch);
	printf("output matrix:\n");
	printGPUMatrix(dA1, l1, batch);

	float* dA0 = l->backward(dA1, batch);
	printf("output matrix:\n");
	printGPUMatrix(dA0, feature, batch);
	
	printf("\ngradient update\n");
	s->gradientUpdate(1);
	l->gradientUpdate(1);

	s->freeMemory();
	l->freeMemory();
	cudaFree(Y);
}*/
