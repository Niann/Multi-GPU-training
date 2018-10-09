#include "model.h"
#include "mnist.h"

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
	cout << "layer number: " << layers.size() << endl;
}

void Model::train(vector<vector<float>> data,vector<int> label, bool print) {
	if (data[0].size() != this->feature_size) {
		cout << data[0].size() << endl;
		cout << "data error! \n";
		return;
	}
	float* X = (float *)malloc(data.size() * this->feature_size * sizeof(float));
	float* Y = (float *)malloc(data.size() * this->out_size * sizeof(float));
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
	if (print) {
		printMatrix(X, 10, data.size());
		printMatrix(Y, 10, data.size());
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
		this->train(batch_data, batch_label, false);
		cout << i/this->batch_size <<" of "<<data.size()/this->batch_size<< "\r";
	}
	cout << "epoch done" << endl;
	this->accuracy(data, label);
}

float Model::accuracy(vector<vector<float>> &data,vector<int> &label) {

	float* X = (float *)malloc(data.size() * this->feature_size * sizeof(float));
	for (int i = 0; i < data.size(); i++) {
		for (int j = 0; j < this->feature_size; j++) {
			X[IDX2C(j, i, this->feature_size)] = data[i][j];
		}
	}

	for (int i = 0; i < this->layers.size(); i++) {
		X = this->layers[i]->forward(X, data.size());
	}
	int count = 0;
	//for (int i = 0; i < data.size()/200; i++) {
	for (int i = 0; i < data.size(); i++) {
		/*
		for (int j = 0; j < this->out_size; j++) {
			cout << *(X + i * this->out_size + j);
		}
		cout << endl;*/
		float* Y1 = X + i * this->out_size;
		float* Y2 = X + (i + 1) * this->out_size;
		//cout << (max_element(Y1, Y2) - Y1) << "	" << label[i] << endl;
		if ((max_element(Y1, Y2) - Y1) == label[i]) {
			//cout << (max_element(Y1, Y2) - Y1) << "	" << label[i] << endl;
			count++;
		}
	}
	float acc = (float)count / (float)data.size();
	cout << "accuracy: " << acc<<"	"<<count<< endl;
	return acc;
}

int main() {
	string filename = "t10k-images-idx3-ubyte";
	int number_of_images = 10000;
	int image_size = 28 * 28;


	//read MNIST iamge into double vector
	vector<vector<float> > train_X;
	read_Mnist(filename, train_X);
	cout << train_X.size() << endl;
	//for (int i=0;i<700;i++)
	//	cout << train_X[0][i] << "	";

	string filename1 = "t10k-labels-idx1-ubyte";

	//read MNIST label into double vector
	vector<int> train_y(number_of_images);
	read_Mnist_Label(filename1, train_y);

	cout << train_y.size() << endl;
	cout << "data loaded" << endl;

	vector<int> layers;
	layers.push_back(200);

	Model* model = new Model(image_size, 10, 0.1f, 32, layers);
	for (int i = 0; i < 10; i++) {
		cout << "start for epoch: " << i << endl;
		model->epoch(train_X, train_y);
	}
	return 0;
}
