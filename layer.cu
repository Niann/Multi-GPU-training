#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <random>

#include "cublas_v2.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define IDX2C(i, j, ld) ((( j )*( ld ))+( i )) // ld - leading dimension

std::default_random_engine generator;
std::normal_distribution<float> distribution(0.0, 0.5);

void initialization(float* a, int size) {
	for (int i = 0; i < size; i++) {
		a[i] = distribution(generator);
		//a[i] = i;
	}
}

__global__ void reluHelper(float* Z, float* dZ, int numElements) {
	// perform relu activation and calculate gradients simultaneously
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements) {
		if (Z[i] < 0) {
			Z[i] = 0;
			dZ[i] = 0;
		}
		else {
			dZ[i] = 1;
		}
	}
}

__global__ void elementMulHelper(float* A, float* B, int numElements, bool invB) {
	// perform relu activation and calculate gradients simultaneously
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements) {
		if (invB) {
			A[i] /= B[i];
		}
		else {
			A[i] *= B[i];
		}
	}
}

__global__ void expHelper(float* A, int numElements) {
	// perform element-wise exp
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements) {
		A[i] = exp(A[i]);
	}
}

__global__ void elementAddHelper(float* A, float* B, float alpha, int numElements) {
	// perform element-wise addition
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements) {
		A[i] += alpha * B[i];
	}
}

void printMatrix(float* a, int r, int c) {
	// print matrix row by row, debugging purpose
	// r - number of rows
	// c - number of columns
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			printf(" %6.3f", a[IDX2C(i, j, r)]);
		}
		printf("\n");
	}
}

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
		Layer(int l1, int l2) {
			l_prev = l1;
			l_curr = l2;

			// allocate memory
			W  = (float *)malloc(l1 * l2 * sizeof(float));
			dW = (float *)malloc(l1 * l2 * sizeof(float));
			b  = (float *)malloc(l2 * sizeof(float));
			db = (float *)malloc(l2 * sizeof(float));

			// initialize W and b
			initialization(W, l1 * l2);
			initialization(b, l2);
			printf("W matrix:\n");
			printMatrix(W, l_curr, l_prev);
			printf("b:\n");
			printMatrix(b, l_curr, 1);
		}

		float* forward(const float* X_in, int batch) {
			// X_in input matrix, size (l_prev, batch_size), each column is a data point
			A_prev = X_in; // save activation from previous layer for backprop
			float* X_out = WX_b(W, X_in, b, l_curr, batch, l_prev); // X_out = Z = W @ X + b
			printf("Z:\n");
			printMatrix(X_out, l_curr, batch);

			// allocate memory for dZ (gradient of activation) and perform activation
			int numElements = l_curr * batch;
			dZ = (float *)malloc(numElements * sizeof(float));
			relu(numElements, X_out, dZ);
			printf("dZ:\n");
			printMatrix(dZ, l_curr, batch);

			return X_out; // X_out = A = relu(Z)
		}

		float* backward(const float* dA, int batch) {
			// dA input matrix, size (l_curr, batch_size), each column is gradient of a datapoint of current layer
			// dA_prev output matrix
			float* dA_prev;
			int numElements = l_prev * batch;
			dA_prev = (float *)malloc(numElements * sizeof(float));

			// calculate dZ, dW, db, dA_prev
			elementwiseMul(l_curr * batch, dZ, dA, false);                                        // dZ = dL/dA * dA/dZ
			printf("dZ:\n");
			printMatrix(dZ, l_curr, batch);

			matrixMul(dZ, A_prev, dW, l_curr, l_prev, batch, false, true, 1/(float)batch, 0.f); // dW = dL/dZ * dZ/dW = 1/m * dZ @ A_prev.T
			printf("dW:\n");
			printMatrix(dW, l_curr, l_prev);

			reduceSum(dZ, db, l_curr, batch, false);                                              // db = dL/dZ * dZ/db = 1/m * sum(dZ, axis=1)
			printf("db:\n");
			printMatrix(db, l_curr, 1);

			matrixMul(W, dZ, dA_prev, l_prev, batch, l_curr, true, false, 1.0f, 0.f);      // dA_prev = dL/dZ * dZ/dA_prev = W.T @ dZ

			return dA_prev;
		}

		void gradientUpdate(float alpha) {
			// perform parameter update w.r.t to gradient direction with learning rate alpha
			elementwiseAdd(l_curr * l_prev, W, dW, -alpha);
			elementwiseAdd(l_curr, b, db, -alpha);

			printf("W matrix:\n");
			printMatrix(W, l_curr, l_prev);
			printf("b:\n");
			printMatrix(b, l_curr, 1);
		}

		void freeMemory() {
			// release memory
			free(W);
			free(dW);
			free(b);
			free(db);
			free(dZ);
			free((float*)A_prev);
		}

	protected:
		// helper functions
		float* WX_b(const float* W, const float* X, const float* b, int m, int n, int k) {
			// perform W @ X + b in a batch
			// m - l_curr, n - batch_size, k - l_prev
			// W is matrix of size (m, k)
			// X is matrix of size (k, n)
			// b is vector of size (m,)
			// c is matrix of size (m, n)
			float * c;
			c = (float *)malloc(m * n * sizeof(float)); // allocate memory for c
			broadcast(c, b, m, n, true); // broadcast b

			matrixMul(W, X, c, m, n, k, false, false, 1.0f, 1.0f);

			return c;
		}

		void matrixMul(const float* A, const float* B, float* C, int m, int n, int k, bool transA, bool transB, float alpha, float beta) {
			// C = op(A) @ op(B) + beta * C
			// op(A) is matrix of size (m, k)
			// op(B) is matrix of size (k, n)
			//   C   is matrix of size (m, n)
			// modifies content of C in-place

			cublasHandle_t handle; // CUBLAS context

			float * d_a; // d_a - a on the device
			float * d_b; // d_b - b on the device
			float * d_c; // d_c - c on the device

			cudaMalloc((void **)& d_a, m*k * sizeof(*A)); // device
			cudaMalloc((void **)& d_b, k*n * sizeof(*B)); // device
			cudaMalloc((void **)& d_c, m*n * sizeof(*C)); // device

			cublasCreate(&handle); // initialize CUBLAS context

			// copy matrices from the host to the device
			cublasSetMatrix(m, k, sizeof(*A), A, m, d_a, m); //a -> d_a
			cublasSetMatrix(k, n, sizeof(*B), B, k, d_b, k); //b -> d_b
			cublasSetMatrix(m, n, sizeof(*C), C, m, d_c, m); //c -> d_c

			// matrix - matrix multiplication : d_c = alpha * op(d_a) @ op(d_b) + beta * d_c
			// op(d_a) - m x k matrix , op(d_b) - k x n matrix , d_c - m x n matrix
			// alpha, beta read from argument
			cublasOperation_t opA = CUBLAS_OP_N;
			cublasOperation_t opB = CUBLAS_OP_N;
			int lda = m;
			int ldb = k;
			if (transA) {
				opA = CUBLAS_OP_T;
				lda = k;
			}
			if (transB) {
				opB = CUBLAS_OP_T;
				ldb = n;
			}
			cublasSgemm(handle, opA, opB, m, n, k, &alpha, d_a, lda, d_b, ldb, &beta, d_c, m);

			// copy matrix from device to host
			cublasGetMatrix(m, n, sizeof(*C), d_c, m, C, m); // cp d_c - >c

			cudaFree(d_a); // free device memory
			cudaFree(d_b); // free device memory
			cudaFree(d_c); // free device memory
			cublasDestroy(handle); // destroy CUBLAS context
		}

		void reduceSum(float* A, float* b, int l, int batch, bool columnwise) {
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

		void elementwiseMul(int numElements, float* A, const float* B, bool invB) {
			// element-wise matrix multiplication, store results in A
			// A = A * B
			unsigned int mem_size = numElements * sizeof(*A);

			float * d_A; // d_A - A on the device
			float * d_B; // d_B - B on the device

			cudaMalloc((void **)& d_A, mem_size); // device
			cudaMalloc((void **)& d_B, mem_size); // device

			// copy matrices from host to device
			cudaMemcpy(d_A, A, mem_size, cudaMemcpyHostToDevice); // A -> d_A
			cudaMemcpy(d_B, B, mem_size, cudaMemcpyHostToDevice); // B -> d_B

			int threadsPerBlock = 256;
			int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
			elementMulHelper<<<blocksPerGrid, threadsPerBlock >>>(d_A, d_B, numElements, invB);

			// copy A from device to host
			cudaMemcpy(A, d_A, mem_size, cudaMemcpyDeviceToHost); // d_A -> A

			cudaFree(d_A); // free device memory
			cudaFree(d_B); // free device memory
		}

		void elementwiseAdd(int numElements, float* A, float* B, float alpha) {
			// element-wise matrix/vector addtion
			// A = A + alpha * B
			unsigned int mem_size = numElements * sizeof(*A);

			float * d_A; // d_A - A on the device
			float * d_B; // d_B - B on the device

			cudaMalloc((void **)& d_A, mem_size); // device
			cudaMalloc((void **)& d_B, mem_size); // device

			// copy matrices from host to device
			cudaMemcpy(d_A, A, mem_size, cudaMemcpyHostToDevice); // A -> d_A
			cudaMemcpy(d_B, B, mem_size, cudaMemcpyHostToDevice); // B -> d_B

			int threadsPerBlock = 256;
			int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
			elementAddHelper<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, alpha, numElements);

			// copy A from device to host
			cudaMemcpy(A, d_A, mem_size, cudaMemcpyDeviceToHost); // d_A -> A

			cudaFree(d_A); // free device memory
			cudaFree(d_B); // free device memory
		}

		void relu(int numElements, float* Z, float* dZ) {
			// perform relu activation and calculate gradients simultaneously
			unsigned int mem_size = numElements * sizeof(*Z);

			float * d_Z;  // d_Z  - Z on the device
			float * d_dZ; // d_dZ - dZ on the device

			cudaMalloc((void **)& d_Z, mem_size);  // device
			cudaMalloc((void **)& d_dZ, mem_size); // device

			// copy Z from host to device
			cudaMemcpy(d_Z, Z, mem_size, cudaMemcpyHostToDevice); // Z -> d_Z

			int threadsPerBlock = 256;
			int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
			reluHelper<<<blocksPerGrid, threadsPerBlock>>>(d_Z, d_dZ, numElements);

			// copy Z from device to host
			cudaMemcpy(Z, d_Z, mem_size, cudaMemcpyDeviceToHost);   // d_Z  -> Z (A)
			cudaMemcpy(dZ, d_dZ, mem_size, cudaMemcpyDeviceToHost); // d_dZ -> dZ

			cudaFree(d_Z);  // free device memory
			cudaFree(d_dZ); // free device memory
		}

		void broadcast(float* c, const float* b, int l, int batch, bool row) {
			// broadcast bias in a batch
			// c - output matrix of size (l, batch_size)
			// b - vector of size (l, 1) if row
			//     vector of size (batch, 1) if column
			if (row) {
				for (int i = 0; i < l; i++) {
					for (int j = 0; j < batch; j++) {
						c[IDX2C(i, j, l)] = b[i];
					}
				}
			}
			else {
				for (int i = 0; i < l; i++) {
					for (int j = 0; j < batch; j++) {
						c[IDX2C(i, j, l)] = b[j];
					}
				}
			}
			printf("initial c matrix:\n");
			printMatrix(c, l, batch);
		}
};

class SoftmaxLayer: public Layer{
	float* P;

    public:
		SoftmaxLayer(int l1, int l2) : Layer(l1, l2) {}

		float* forward(const float* X_in, int batch) {
			// X_in input matrix, size (l_prev, batch_size), each column is a data point
			A_prev = X_in; // save activation from previous layer for backprop
			float* X_out = WX_b(W, X_in, b, l_curr, batch, l_prev); // X_out = Z = W @ X + b
			printf("Z:\n");
			printMatrix(X_out, l_curr, batch);

			// allocate memory for P (prob matrix) and perform activation
			int numElements = l_curr * batch;
			P = (float *)malloc(numElements * sizeof(float));
			softmax(numElements, X_out);
			P = X_out; // store P for backprop

			return X_out; // X_out = softmax(Z)
		}

	private:
		void softmax(int numElements, float* Z) {
			// softmax operation over each coloum of Z
			// store gradients in dZ
			// 1st x = exp(x) for each element x in Z
			// 2nd p = sum(x) for each column in Z
			// 3rd x = x/p for each element each column in Z
			float* d_Z;
			int batch = numElements / l_curr;
			unsigned int mem_size = numElements * sizeof(*Z);

			// allocate memory for Z on device
			cudaMalloc((void **)& d_Z, mem_size);
			cudaMemcpy(d_Z, Z, mem_size, cudaMemcpyHostToDevice); // Z -> d_Z

			// 1st x = exp(x) for each element x in Z
			int threadsPerBlock = 256;
			int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
			expHelper<<<blocksPerGrid, threadsPerBlock>>>(d_Z, numElements);

			cudaMemcpy(Z, d_Z, mem_size, cudaMemcpyDeviceToHost);   // d_Z  -> Z
			cudaFree(d_Z);  // free device memory
			printf("exp:\n");
			printMatrix(Z, l_curr, batch);

			// 2nd p = sum(x) for each column in Z
			float* p; // vector length batch_size
			p = (float *)malloc(batch * sizeof(float));
			reduceSum(Z, p, l_curr, batch, true);
			printf("p:\n");
			printMatrix(p, batch, 1);

			// 3rd x = x/p for each element each column in Z
			float * P; // probability matrix of size (l_curr, batch)
			P = (float *)malloc(numElements * sizeof(float));
			broadcast(P, p, l_curr, batch, false);
			elementwiseMul(numElements, Z, P, true);
			free(p);
			free(P);
		}
};

int main() {
	int batch = 5;
	int feature = 4;
	int l1 = 3;
	int l2 = 2;

	float* X;
	X = (float *)malloc(feature * batch * sizeof(float));
	initialization(X, feature * batch);
	printf("foward pass\n");
	printf("input matrix:\n");
	printMatrix(X, feature, batch);

	Layer l = Layer(feature, l1);
	SoftmaxLayer s = SoftmaxLayer(l1, l2);
	float* X1 = l.forward(X, batch);
	printf("output matrix:\n");
	printMatrix(X1, l1, batch);

	float* X2 = s.forward(X1, batch);
	printf("output matrix:\n");
	printMatrix(X2, l2, batch);
	/*
	printf("\nbackward pass\n");
	float* dA;
	dA = (float *)malloc(l1 * batch * sizeof(float));
	initialization(dA, l1 * batch);
	printf("input matrix:\n");
	printMatrix(dA, l1, batch);

	float* dA0 = l.backward(dA, batch);
	printf("output matrix:\n");
	printMatrix(dA0, feature, batch);

	printf("\ngradient updata\n");
	l.gradientUpdate(1);
	l.freeMemory(); */
}