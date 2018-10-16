#include "layer_cpu.h"

#define IDX2C(i, j, ld) ((( j )*( ld ))+( i )) // ld - leading dimension

namespace cpu {

	std::default_random_engine generator;
	std::normal_distribution<float> distribution(0.0f, 0.005f);

	void initialization(float* a, int size, bool t) {
		for (int i = 0; i < size; i++) {
			if (!t)
				a[i] = distribution(generator);
			else
				a[i] = 1;
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
		for (int i = 0; i < numElements; i++) {
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
		// perform element-wise multiplication
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

	void matrixMulHelper(const float* A, const float* B, float* C, int m, int n, int k, float alpha, float beta) {
		// perform matrix-matrix multiplication
		int i, j, k_;
//#pragma omp parallel shared(A,B,C) private(i,j,k)
		{
//#pragma omp for schedule(dynamic)
			for (i = 0; i < m; i++)
			{
				for (j = 0; j < n; j++)
				{
					C[IDX2C(i, j, m)] *= beta;
					for (k_ = 0; k_ < k; k_++)
					{
						C[IDX2C(i, j, m)] += alpha * A[IDX2C(i, k_, m)] * B[IDX2C(k_, j, k)];
					}
				}
			}
		}
	}

	void transpose(float* A, const float *B, int m, int n) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				A[IDX2C(j, i, n)] = B[IDX2C(i, j, m)];
			}
		}
	}

	void copyMat(float* A, const float *B, int numElements) {
		for (int i = 0; i < numElements; i++) {
			A[i] = B[i];
		}
	}
}

using namespace cpu;

Layer_cpu::Layer_cpu(int l1, int l2) {
	l_prev = l1;
	l_curr = l2;

	// allocate memory
	W = (float *)malloc(l1 * l2 * sizeof(float));
	dW = (float *)malloc(l1 * l2 * sizeof(float));
	b = (float *)malloc(l2 * sizeof(float));
	db = (float *)malloc(l2 * sizeof(float));

	// initialize W and b
	initialization(W, l1 * l2, false);
	initialization(b, l2, false);
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
	float * c;
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
	
	matrixMulHelper(tA, tB, C, m, n, k, alpha, beta);
	free(tA);
	free(tB);
}

void Layer_cpu::reduceSum(float* A, float* b, int l, int batch, bool columnwise) {
	// reduce sum row-wise or column-wise
	// store results in b
	// get b by matrix - vector multiplication

	// create a vector of same size as b filled with 1
	float* x;
	float alpha;
	if (columnwise) {
		x = (float *)malloc(l * sizeof(float));
		alpha = 1.0f;
		for (int i = 0; i < l; i++)
			x[i] = 1;

		matrixMul(A, x, b, batch, 1, l, true, false, alpha, 0.f);
	}
	else {
		x = (float *)malloc(batch * sizeof(float));
		alpha = 1 / (float)batch;
		for (int i = 0; i < batch; i++)
			x[i] = 1;

		matrixMul(A, x, b, l, 1, batch, false, false, alpha, 0.f);
	}
	free(x);
}

void Layer_cpu::elementwiseMul(int numElements, float* A, const float* B, bool invB) {
	// element-wise matrix multiplication, store results in A
	// A = A * B
	elementMulHelper(A, B, numElements, invB);
}

void Layer_cpu::elementwiseAdd(int numElements, float* A, const float* B, float alpha) {
	// element-wise matrix/vector addtion
	// A = A + alpha * B
	elementAddHelper(A, B, alpha, numElements);
}

void Layer_cpu::elementwiseExp(float* A, int numElements) {
	// element-wise exponential
	expHelper(A, numElements);
}

void Layer_cpu::broadcast(float* A, float* b, int r, int c, bool row) {
	// broadcast b to A by row/column
	broadcastHelper(A, b, r, c, row);
}

void ReluLayer_cpu::relu(int numElements, float* Z, float* dZ) {
	// perform relu activation and calculate gradients simultaneously
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
	using namespace cpu;
	int batch = 5;
	int feature = 10;
	int l1 = 6;
	int l2 = 5;

	float* X;
	float* Y;
	X = (float *)malloc(feature * batch * sizeof(float));
	initialization(X, feature * batch, true);
	Y = (float *)malloc(l2 * batch * sizeof(float));
	initialization(Y, l2 * batch, true);
	printf("foward pass\n");
	printf("input matrix:\n");
	printMatrix(X, feature, batch);

	Layer_cpu* l = new ReluLayer_cpu(feature, l1);
	Layer_cpu* s = new SoftmaxLayer_cpu(l1, l2);
	float* X1 = l->forward(X, batch);
	printf("output matrix:\n");
	printMatrix(X1, l1, batch);

	float* X2 = s->forward(X1, batch);
	printf("output matrix:\n");
	printMatrix(X2, l2, batch);
	
	printf("\nbackward pass\n");
	printf("input matrix:\n");
	printMatrix(Y, l2, batch);

	float* dA1 = s->backward(Y, batch);
	printf("output matrix:\n");
	printMatrix(dA1, l1, batch);

	float* dA0 = l->backward(dA1, batch);
	printf("output matrix:\n");
	printMatrix(dA0, feature, batch);
	
	printf("\ngradient update\n");
	s->SGDUpdate(1);
	l->SGDUpdate(1);

	s->freeMemory();
	l->freeMemory();
	free(Y);
}*/
