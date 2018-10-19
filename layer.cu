#include "layer.cuh"

#define IDX2C(i, j, ld) ((( j )*( ld ))+( i )) // ld - leading dimension

void initialization(float* a, int size) {
	std::default_random_engine generator;
	std::normal_distribution<float> distribution(0.0f, 0.005f);
	for (int i = 0; i < size; i++) {
		// use a universal hash function
		// guarantee that same size of input has same initial value
		generator.seed(((i * 233 + 66666699) % 433494437) % (size * 10));
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

void printGPUMatrix(const float* a, int r, int c) {
	// print matrix on gpu
	float* h = (float *)malloc(r * c * sizeof(float));
	cudaMemcpy(h, a, r * c * sizeof(float), cudaMemcpyDeviceToHost);
	printMatrix(h, r, c);
	free(h);
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

__global__ void broadcastHelper(float* A, const float* b, int r, int c, bool row) {
	// broadcast b to A by row/column
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < r * c) {
		if (row) {
			A[i] = b[i % r];
		}
		else {
			A[i] = b[i / r];
		}
	}
}

__global__ void elementMulHelper(float* A, const float* B, int numElements, bool invB) {
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

__global__ void elementAddHelper(float* A, const float* B, float alpha, int numElements) {
	// perform element-wise addition
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements) {
		A[i] += alpha * B[i];
	}
}

__global__ void elementAvgHelper(float* dest, float* src, int numElements, int copy) {
	// element-wise average
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements) {
		float temp = 0;
		for (int j = 0; j < copy; j++) {
			temp += src[numElements * j + i];
		}
		dest[i] = temp / copy;
	}
}

Layer::Layer(int l1, int l2, int gpu) {
	l_prev = l1;
	l_curr = l2;
	gpuNum = gpu;

	// allocate memory
	float* h_W = (float *)malloc(l1 * l2 * sizeof(float));
	float* h_b = (float *)malloc(l2 * sizeof(float));
	cudaMalloc((void **)& W, l1 * l2 * sizeof(float));
	cudaMalloc((void **)& dW, l1 * l2 * sizeof(float));
	cudaMalloc((void **)& b, l2 * sizeof(float));
	cudaMalloc((void **)& db, l2 * sizeof(float));

	// initialize W and b
	initialization(h_W, l1 * l2);
	initialization(h_b, l2);
	cudaMemcpy(W, h_W, l1 * l2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b, h_b, l2 * sizeof(float), cudaMemcpyHostToDevice);
	free(h_W);
	free(h_b);
}

ReluLayer::ReluLayer(int l1, int l2, int gpu) : Layer(l1, l2, gpu) {}

SoftmaxLayer::SoftmaxLayer(int l1, int l2, int gpu) : Layer(l1, l2, gpu) {}

float* ReluLayer::forward(float* X_in, int batch, bool inference) {
	// X_in input matrix, size (l_prev, batch_size), each column is a data point
	A_prev = X_in; // save activation from previous layer for backprop
	float* X_out = WX_b(W, X_in, b, l_curr, batch, l_prev); // X_out = Z = W @ X + b

	// allocate memory for dZ (gradient of activation) and perform activation
	int numElements = l_curr * batch;
	cudaMalloc((void **)& dZ, numElements * sizeof(float));
	relu(numElements, X_out, dZ);

	if (inference){
		cudaFree(A_prev);
	}

	return X_out; // X_out = A = relu(Z)
}

float* SoftmaxLayer::forward(float* X_in, int batch, bool inference) {
	// X_in input matrix, size (l_prev, batch_size), each column is a data point
	A_prev = X_in; // save activation from previous layer for backprop
	float* X_out = WX_b(W, X_in, b, l_curr, batch, l_prev); // X_out = Z = W @ X + b

	// allocate memory for dZ and perform activation
	int numElements = l_curr * batch;
	softmax(numElements, X_out);
	dZ = X_out; // store for backprop

	if (inference){
		cudaFree(A_prev);
	}

	return X_out; // X_out = softmax(Z)
}

float* ReluLayer::backward(float* dA, int batch) {
	// dA input matrix, size (l_curr, batch_size), each column is gradient of a datapoint of current layer
	// dA_prev output matrix
	float* dA_prev;
	int numElements = l_prev * batch;
	cudaMalloc((void **)& dA_prev, numElements * sizeof(float));

	// calculate dZ, dW, db, dA_prev
	elementwiseMul(l_curr * batch, dZ, dA, false);                                        // dZ = dL/dA * dA/dZ
	matrixMul(dZ, A_prev, dW, l_curr, l_prev, batch, false, true, 1 / (float)batch, 0.f); // dW = dL/dZ * dZ/dW = 1/m * dZ @ A_prev.T
	reduceSum(dZ, db, l_curr, batch, false);                                              // db = dL/dZ * dZ/db = 1/m * sum(dZ, axis=1)
	matrixMul(W, dZ, dA_prev, l_prev, batch, l_curr, true, false, 1.0f, 0.f);             // dA_prev = dL/dZ * dZ/dA_prev = W.T @ dZ

	cudaFree(dA);
	cudaFree(dZ);
	return dA_prev;
}

float* SoftmaxLayer::backward(float* Y, int batch) {
	// y - one-hot matrix of size (l_curr, batch)
	// each column is a one_hot label for corresponding forward data
	// dA_prev out_put matrix - gradient of activation from previous layer
	float* dA_prev;
	int numElements = l_prev * batch;
	cudaMalloc((void **)& dA_prev, numElements * sizeof(float));

	// calculate dZ, dW, db, dA_prev
	elementwiseAdd(l_curr * batch, dZ, Y, -1.0f);                                         // dZ = P - Y
	matrixMul(dZ, A_prev, dW, l_curr, l_prev, batch, false, true, 1 / (float)batch, 0.f); // dW = 1/m * dZ @ A_prev.T
	reduceSum(dZ, db, l_curr, batch, false);                                              // db = 1/m * sum(dZ, axis=1)
	matrixMul(W, dZ, dA_prev, l_prev, batch, l_curr, true, false, 1.0f, 0.f);             // dA = W.T @ dZ

	cudaFree(dZ);
	return dA_prev;
}

void Layer::SGDUpdate(float alpha) {
	// perform parameter update w.r.t to gradient direction with learning rate alpha
	MPI_Barrier(MPI_COMM_WORLD);

	// allocate gpu memory for dW and db on all devices
	float* all_dW;
	float* all_db;
	cudaMalloc((void **)& all_dW, l_prev * l_curr * gpuNum* sizeof(float));
	cudaMalloc((void **)& all_db, l_curr * gpuNum * sizeof(float));

	// synchonize gradients from all devices
	MPI_Allgather(dW, l_prev * l_curr, MPI_FLOAT, all_dW, l_prev * l_curr, MPI_FLOAT, MPI_COMM_WORLD);
	MPI_Allgather(db, l_curr, MPI_FLOAT, all_db, l_curr, MPI_FLOAT, MPI_COMM_WORLD);

	// calculate mean of all_dW and all_db and store in dW and db
	averageGradients(dW, all_dW, db, all_db);

	// update parameters
	elementwiseAdd(l_curr * l_prev, W, dW, -alpha);
	elementwiseAdd(l_curr, b, db, -alpha);

	cudaFree(all_dW);
	cudaFree(all_db);
}

void Layer::freeMemory() {
	// release memory
	cudaFree(W);
	cudaFree(dW);
	cudaFree(b);
	cudaFree(db);
}


// helper functions
float* Layer::WX_b(const float* W, const float* X, float* b, int m, int n, int k) {
	// perform W @ X + b in a batch
	// m - l_curr, n - batch_size, k - l_prev
	// W is matrix of size (m, k)
	// X is matrix of size (k, n)
	// b is vector of size (m,)
	// c is matrix of size (m, n)
	float * c; // c - c on the device
	cudaMalloc((void **)& c, m*n * sizeof(*c)); // device

	broadcast(c, b, m, n, true); // broadcast b

	matrixMul(W, X, c, m, n, k, false, false, 1.0f, 1.0f);

	return c;
}

void Layer::matrixMul(const float* A, const float* B, float* C, int m, int n, int k, bool transA, bool transB, float alpha, float beta) {
	// C = op(A) @ op(B) + beta * C
	// op(A) is matrix of size (m, k)
	// op(B) is matrix of size (k, n)
	//   C   is matrix of size (m, n)
	// modifies content of C in-place

	cublasHandle_t handle; // CUBLAS context

	cublasCreate(&handle); // initialize CUBLAS context

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
	cublasSgemm(handle, opA, opB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, m);

	cublasDestroy(handle); // destroy CUBLAS context
}

void Layer::reduceSum(float* A, float* b, int l, int batch, bool columnwise) {
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
		cudaMalloc((void **)& d_x, l * sizeof(float));
		cudaMemcpy(d_x, x, l * sizeof(float), cudaMemcpyHostToDevice);

		matrixMul(A, d_x, b, batch, 1, l, true, false, alpha, 0.f);
	}
	else {
		x = (float *)malloc(batch * sizeof(float));
		alpha = 1 / (float)batch;
		for (int i = 0; i < batch; i++)
			x[i] = 1;
		cudaMalloc((void **)& d_x, batch * sizeof(float));
		cudaMemcpy(d_x, x, batch * sizeof(float), cudaMemcpyHostToDevice);

		matrixMul(A, d_x, b, l, 1, batch, false, false, alpha, 0.f);
	}
	free(x);
	cudaFree(d_x);
}

void Layer::elementwiseMul(int numElements, float* A, const float* B, bool invB) {
	// element-wise matrix multiplication, store results in A
	// A = A * B
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	elementMulHelper<<<blocksPerGrid, threadsPerBlock>>>(A, B, numElements, invB);
}

void Layer::elementwiseAdd(int numElements, float* A, const float* B, float alpha) {
	// element-wise matrix/vector addtion
	// A = A + alpha * B
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	elementAddHelper<<<blocksPerGrid, threadsPerBlock>>>(A, B, alpha, numElements);
}

void Layer::elementwiseExp(float* A, int numElements) {
	// element-wise exponential
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	expHelper<<<blocksPerGrid, threadsPerBlock >>>(A, numElements);
}

void Layer::broadcast(float* A, float* b, int r, int c, bool row) {
	// broadcast b to A by row/column
	int threadsPerBlock = 256;
	int blocksPerGrid = (r * c + threadsPerBlock - 1) / threadsPerBlock;
	broadcastHelper<<<blocksPerGrid, threadsPerBlock>>>(A, b, r, c, row);
}

void Layer::averageGradients(float* dW, float* all_dW, float* db, float* all_db) {
	// average dW and db
	int threadsPerBlock = 256;

	int numElements_W = l_curr * l_prev;
	int blocksPerGrid_W = (numElements_W + threadsPerBlock - 1) / threadsPerBlock;
	elementAvgHelper<<<blocksPerGrid_W, threadsPerBlock>>>(dW, all_dW, numElements_W, gpuNum);

	int numElements_b = l_curr;
	int blocksPerGrid_b = (numElements_b + threadsPerBlock - 1) / threadsPerBlock;
	elementAvgHelper<<<blocksPerGrid_b, threadsPerBlock>>>(db, all_db, numElements_b, gpuNum);
}

void ReluLayer::relu(int numElements, float* Z, float* dZ) {
	// perform relu activation and calculate gradients simultaneously
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	reluHelper<<<blocksPerGrid, threadsPerBlock>>>(Z, dZ, numElements);
}

void SoftmaxLayer::softmax(int numElements, float* Z) {
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
	cudaMalloc((void **)& p, batch * sizeof(float));
	reduceSum(Z, p, l_curr, batch, true);

	// 3rd x = x/p for each element each column in Z
	float * P;
	cudaMalloc((void **)& P, numElements * sizeof(float));
	broadcast(P, p, l_curr, batch, false);
	elementwiseMul(numElements, Z, P, true);

	cudaFree(p);
	cudaFree(P);
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
