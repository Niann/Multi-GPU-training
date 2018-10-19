#include "model.h"
#include "mnist.h"
#include <time.h>

#define INPUT_SIZE 784
#define LABEL_SIZE 10
#define BATCH_SIZE 256
#define EPOCH 10

#define IDX2C(i, j, ld) ((( j )*( ld ))+( i ))

Model::Model(int in, int out, float lr, int batch_size, vector<int> layer_size, int gpu) {
	this->feature_size = in;
	this->out_size = out;
	this->batch_size = batch_size;
	this->learning_rate = lr;

	layers.push_back(new ReluLayer(in, layer_size[0], gpu));
	for (int i = 0; i < layer_size.size() - 1; i++) {
		layers.push_back(new ReluLayer(layer_size[i], layer_size[i+1], gpu));
	}
	layers.push_back(new SoftmaxLayer(layer_size.back(), out, gpu));

	// allocate memory for place holders
	X = (float *)malloc(batch_size * in * sizeof(float));
	cudaMalloc((void **)& d_X, batch_size * in * sizeof(float));
	Y = (float *)malloc(batch_size * out * sizeof(float));
	cudaMalloc((void **)& d_Y, batch_size * out * sizeof(float));

	cout << "layer number: " << layers.size() << endl;
}

void Model::train(vector<vector<float>> data,vector<int> label) {
	if (data[0].size() != this->feature_size) {
		cout << data[0].size() << endl;
		cout << "data error! \n";
		return;
	}

	for (int i = 0; i < data.size(); i++) {
		for (int j = 0; j < this->feature_size; j++) {
			X[IDX2C(j, i, this->feature_size)] = data[i][j];
		}
	}

	for (int i = 0; i < label.size(); i++) {
		for (int j = 0; j < this->out_size; j++) {
			Y[IDX2C(j, i, this->out_size)] = 0;
		}
		Y[IDX2C(label[i], i, this->out_size)] = 1;
	}

	// forward pass
	cudaMemcpy(d_X, X, data.size() * feature_size * sizeof(float), cudaMemcpyHostToDevice);
	float* X_in = d_X;
	for (int i = 0; i < this->layers.size(); i++) {
		X_in = layers[i]->forward(X_in, batch_size, false);
	}

	// backward pass
	cudaMemcpy(d_Y, Y, data.size() * out_size * sizeof(float), cudaMemcpyHostToDevice);
	float* Y_in = d_Y;
	for (int i = this->layers.size()-1; i >=0 ; i--) {
		Y_in = layers[i]->backward(Y_in, batch_size);
	}
	cudaFree(Y_in);

	// parameter update
	for (int i = 0; i < this->layers.size(); i++) {
		layers[i]->SGDUpdate(this->learning_rate);
	}
}

void Model::epoch(vector<vector<float>> &data, vector<int> &label) {
	for (int i = 0; i <= data.size() - batch_size; i += batch_size) {
		vector<vector<float>> batch_data;
		vector<int> batch_label;
		for (int j = 0; j < batch_size; j++) {
			batch_data.push_back(data[i + j]);
			batch_label.push_back(label[i + j]);
		}
		this->train(batch_data, batch_label);
		//cout << i/this->batch_size <<" of "<<data.size()/this->batch_size<< "\r";
	}
	cout << "epoch done" << endl;
}

float Model::accuracy(vector<vector<float>> &data,vector<int> &label) {
	float* X_test = (float *)malloc(data.size() * feature_size * sizeof(float));
	for (int i = 0; i < data.size(); i++) {
		for (int j = 0; j < feature_size; j++) {
			X_test[IDX2C(j, i, feature_size)] = data[i][j];
		}
	}

	float* d_X_test;
	cudaMalloc((void **)& d_X_test, data.size() * feature_size * sizeof(float));
	cudaMemcpy(d_X_test, X_test, data.size() * feature_size * sizeof(float), cudaMemcpyHostToDevice);

	for (int i = 0; i < this->layers.size(); i++) {
		d_X_test = this->layers[i]->forward(d_X_test, data.size(), true);
	}

	float* preds = (float *)malloc(data.size() * out_size * sizeof(float));
	cudaMemcpy(preds, d_X_test, data.size() * out_size * sizeof(float), cudaMemcpyDeviceToHost);

	int count = 0;
	for (int i = 0; i < data.size(); i++) {
		float* Y1 = preds + i * this->out_size;
		float* Y2 = preds + (i + 1) * this->out_size;
		if ((max_element(Y1, Y2) - Y1) == label[i]) {
			count++;
		}
	}

	free(X_test);
	free(preds);

	float acc = (float)count / (float)data.size();
	cout << "accuracy: " << acc << endl;
	return acc;
}

void Model::freeMemory() {
	for (int i = 0; i < this->layers.size(); i++) {
		this->layers[i]->freeMemory();
		delete this->layers[i];
	}
	cudaFree(X);
	cudaFree(Y);
}


int main(int argc, char **argv) {
	int comm_size;
	int rank;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	string Xtrain_file = "train-images-idx3-ubyte";
	string Ytrain_file = "train-labels-idx1-ubyte";
	string Xtest_file = "t10k-images-idx3-ubyte";
	string Ytest_file = "t10k-labels-idx1-ubyte";

	//read MNIST iamge into float vector
	vector<vector<float>> train_X;
	vector<vector<float>> test_X;
	read_Mnist(Xtrain_file, train_X, rank, comm_size);
	read_Mnist(Xtest_file, test_X, 0, 1);
	cout << "training set size: " << train_X.size() << endl;
	cout << "test set size: " << test_X.size() << endl;

	//read MNIST label into int vector
	vector<int> train_y;
	vector<int> test_y;
	read_Mnist_Label(Ytrain_file, train_y, rank, comm_size);
	read_Mnist_Label(Ytest_file, test_y, 0, 1);
	cout << train_y.size() << endl;
	cout << test_y.size() << endl;
	cout << "data loaded" << endl;

	vector<int> layers;
	layers.push_back(2000);
	layers.push_back(2000);

	clock_t t = clock();
	Model* model = new Model(INPUT_SIZE, LABEL_SIZE, 0.1f, BATCH_SIZE, layers, comm_size);
	for (int i = 0; i < EPOCH; i++) {
		cout << "start for epoch: " << i << endl;
		model->epoch(train_X, train_y);
		//model->epoch(test_X, test_y);
		model->accuracy(test_X, test_y);
	}
	t = clock() - t;
	float time = (float)t / CLOCKS_PER_SEC;
	cout << "time consuming: " << time << " seconds" << endl;
	model->freeMemory();
	MPI_Finalize();

	return 0;
}
