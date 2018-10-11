#include "model.h"
#include "mnist.h"
#include <time.h>

#define IDX2C(i, j, ld) ((( j )*( ld ))+( i )) // ld - leading dimension


Model::Model(int in, int out, float lr, int batch_size, vector<int> layer_size) {
	this->feature_size = in;
	this->out_size = out;
	this->batch_size = batch_size;
	this->learning_rate = lr;

	layers.push_back(new ReluLayer(in, layer_size[0]));
	for (int i = 0; i < layer_size.size() - 1; i++) {
		layers.push_back(new ReluLayer(layer_size[i], layer_size[i+1]));
	}
	layers.push_back(new SoftmaxLayer(layer_size.back(), out));

	X = (float *)malloc(batch_size * feature_size * sizeof(float));
	Y = (float *)malloc(batch_size * out * sizeof(float));

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

	for (int i = 0; i < this->layers.size(); i++) {
		X = layers[i]->forward(X, data.size());
	}

	for (int i = this->layers.size()-1; i >=0 ; i--) {
		Y = layers[i]->backward(Y, data.size());
	}
	for (int i = 0; i < this->layers.size(); i++) {
		layers[i]->gradientUpdate(this->learning_rate);
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
		cout << i/this->batch_size <<" of "<<data.size()/this->batch_size<< "\r";
	}
	cout << "epoch done" << endl;
}

float Model::accuracy(vector<vector<float>> &data,vector<int> &label) {

	float* X_test = (float *)malloc(data.size() * this->feature_size * sizeof(float));
	for (int i = 0; i < data.size(); i++) {
		for (int j = 0; j < this->feature_size; j++) {
			X_test[IDX2C(j, i, this->feature_size)] = data[i][j];
		}
	}

	for (int i = 0; i < this->layers.size(); i++) {
		X_test = this->layers[i]->forward(X_test, data.size());
	}

	int count = 0;
	for (int i = 0; i < data.size(); i++) {
		float* Y1 = X_test + i * this->out_size;
		float* Y2 = X_test + (i + 1) * this->out_size;
		if ((max_element(Y1, Y2) - Y1) == label[i]) {
			count++;
		}
	}
	free(X_test);
	float acc = (float)count / (float)data.size();
	cout << "accuracy: " << acc<<"	"<<count<< endl;
	return acc;
}

int main() {
	string Xtrain_file = "train-images-idx3-ubyte";
	string Ytrain_file = "train-labels-idx1-ubyte";
	string Xtest_file = "t10k-images-idx3-ubyte";
	string Ytest_file = "t10k-labels-idx1-ubyte";
	int image_size = 28 * 28;

	//read MNIST iamge into float vector
	vector<vector<float>> train_X;
	vector<vector<float>> test_X;
	read_Mnist(Xtrain_file, train_X);
	read_Mnist(Xtest_file, test_X);
	cout << "training set size: " << train_X.size() << endl;
	cout << "test set size: " << test_X.size() << endl;

	//read MNIST label into int vector
	vector<int> train_y(train_X.size());
	vector<int> test_y(test_X.size());
	read_Mnist_Label(Ytrain_file, train_y);
	read_Mnist_Label(Ytest_file, test_y);
	cout << train_y.size() << endl;
	cout << test_y.size() << endl;
	cout << "data loaded" << endl;

	vector<int> layers;
	layers.push_back(300);
	layers.push_back(300);

	clock_t t = clock();
	Model* model = new Model(image_size, 10, 0.1f, 64, layers);
	for (int i = 0; i < 10; i++) {
		cout << "start for epoch: " << i << endl;
		model->epoch(train_X, train_y);
		model->accuracy(test_X, test_y);
	}
	t = clock() - t;
	float time = (float)t / CLOCKS_PER_SEC;
	cout << "time consuming: " << time << "seconds" << endl;

	return 0;
}
