#include "model.h"
#include "mnist.h"

#define IDX2C(i, j, ld) ((( i )*( ld ))+( j )) // ld - leading dimension


Model::Model(int in, int out, float lr, int batch_size, vector<int> layer_size) {
	this->feature_size = in;
	this->out_size = out;
	this->batch_size = batch_size;
	this->learning_rate = lr;
	layers.push_back(Layer(in, layer_size[0]));
	for (int i = 0; i < layer_size.size() - 1; i++) {
		layers.push_back(Layer(layer_size[i], layer_size[i]));
	}
	layers.push_back(SoftmaxLayer(layer_size.back(), out));
}

void Model::train(vector<vector<float>> data,vector<int> label) {
	if (data[0].size() != this->feature_size) {
		cout << data[0].size() << endl;
		cout << "data error! \n";
		return;
	}
	float* X = (float *)malloc(data.size() * this->feature_size * sizeof(float));
	float* Y = (float *)malloc(data.size() * this->out_size * sizeof(float));
	for (int i = 0; i < data.size(); i++) {
		for (int j = 0; j < this->feature_size; j++) {
			X[IDX2C(i, j, this->feature_size)] = data[i][j];
		}
	}
	for (int i = 0; i < label.size(); i++) {
		for (int j = 0; j < this->out_size; j++) {
			Y[IDX2C(i, j, this->out_size)] = 0;
		}
		Y[IDX2C(i, label[i], this->out_size)] = 1;
	}

	for (int i = 0; i < this->layers.size(); i++) {
		X = this->layers[i].forward(X, data.size());
	}

	for (int i = this->layers.size()-1; i >=0 ; i--) {
		Y = this->layers[i].backward(Y, data.size());
	}
	for (int i = 0; i < this->layers.size(); i++) {
		this->layers[i].gradientUpdate(this->learning_rate);
	}
}
	
void Model::epoch(vector<vector<float>> &data, vector<int> &label) {
	for (int i = 0; i < data.size() - batch_size; i += batch_size) {
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
	this->accuracy(data, label);
}

float Model::accuracy(vector<vector<float>> &data,vector<int> &label) {
	float* X = (float *)malloc(data.size() * this->feature_size * sizeof(float));
	for (int i = 0; i < data.size(); i++) {
		for (int j = 0; j < this->feature_size; j++) {
			X[IDX2C(i, j, this->feature_size)] = data[i][j];
		}
	}

	for (int i = 0; i < this->layers.size(); i++) {
		X = this->layers[i].forward(X, data.size());
	}
	int count = 0;
	for (int i = 0; i < data.size(); i++) {
		float* Y1 = X + i * this->out_size;
		float* Y2 = X + (i + 1) * this->out_size;
		//cout << (max_element(Y1, Y2) - Y1) << "	" << label[i] << endl;
		if ((max_element(Y1, Y2) - Y1) == label[i]) {
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
	cout << train_X[0].size() << endl;

	string filename1 = "t10k-labels-idx1-ubyte";

	//read MNIST label into double vector
	vector<int> train_y(number_of_images);
	read_Mnist_Label(filename1, train_y);
//	vector<int> train_y(number_of_images);

	//for (int i = 0; i < ttrain_y.size(); i++) {
	//	train_y.push_back((int) ttrain_y[i]);
	//}
	cout << train_y.size() << endl;
	cout << "data loaded" << endl;
	
	vector<int> layers;
	layers.push_back(3);
	layers.push_back(3);

	Model* model = new Model(image_size,10,0.001,32, layers);
	for (int i = 0; i < 10; i++) {
		cout << "start for epoch: " << i << endl;
		model->epoch(train_X, train_y);
	}
	return 0;
}