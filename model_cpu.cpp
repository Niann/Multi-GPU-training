#include "model_cpu.h"

#define IDX2C(i, j, ld) ((( j )*( ld ))+( i ))

Model_cpu::Model_cpu(int in, int out, float lr, int batch_size, vector<int> layer_size) {
	this->feature_size = in;
	this->out_size = out;
	this->batch_size = batch_size;
	this->learning_rate = lr;

	layers.push_back(new ReluLayer_cpu(in, layer_size[0]));
	for (int i = 0; i < layer_size.size() - 1; i++) {
		layers.push_back(new ReluLayer_cpu(layer_size[i], layer_size[i + 1]));
	}
	layers.push_back(new SoftmaxLayer_cpu(layer_size.back(), out));

	// allocate memory for place holders
	X = (float *)malloc(batch_size * in * sizeof(float));
	Y = (float *)malloc(batch_size * out * sizeof(float));

	cout << "layer number: " << layers.size() << endl;
}

void Model_cpu::train(vector<vector<float>> data, vector<int> label) {
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
	float* X_in = X;
	for (int i = 0; i < this->layers.size(); i++) {
		X_in = layers[i]->forward(X_in, batch_size);
	}

	// backward pass
	float* Y_in = Y;
	for (int i = this->layers.size() - 1; i >= 0; i--) {
		Y_in = layers[i]->backward(Y_in, batch_size);
	}
	free(Y_in);

	// parameter update
	for (int i = 0; i < this->layers.size(); i++) {
		layers[i]->SGDUpdate(this->learning_rate);
	}
}

void Model_cpu::epoch(vector<vector<float>> &data, vector<int> &label) {
	for (int i = 0; i <= data.size() - batch_size; i += batch_size) {
		vector<vector<float>> batch_data;
		vector<int> batch_label;
		for (int j = 0; j < batch_size; j++) {
			batch_data.push_back(data[i + j]);
			batch_label.push_back(label[i + j]);
		}
		this->train(batch_data, batch_label);
		cout << i/this->batch_size <<" of "<<data.size()/this->batch_size<< "\r";
	}
	cout << "epoch done" << endl;
}

float Model_cpu::accuracy(vector<vector<float>> &data, vector<int> &label) {
	float* X_test = (float *)malloc(data.size() * feature_size * sizeof(float));
	for (int i = 0; i < data.size(); i++) {
		for (int j = 0; j < feature_size; j++) {
			X_test[IDX2C(j, i, feature_size)] = data[i][j];
		}
	}

	float* X_test_in = X_test;
	for (int i = 0; i < this->layers.size(); i++) {
		X_test_in = this->layers[i]->forward(X_test_in, data.size());
	}

	int count = 0;
	for (int i = 0; i < data.size(); i++) {
		float* Y1 = X_test_in + i * this->out_size;
		float* Y2 = X_test_in + (i + 1) * this->out_size;
		if ((max_element(Y1, Y2) - Y1) == label[i]) {
			count++;
		}
	}

	free(X_test);
	free(X_test_in);

	float acc = (float)count / (float)data.size();
	cout << "accuracy: " << acc << endl;
	return acc;
}

void Model_cpu::freeMemory() {
	for (int i = 0; i < this->layers.size(); i++) {
		this->layers[i]->freeMemory();
		delete this->layers[i];
	}
	free(X);
	free(Y);
}
