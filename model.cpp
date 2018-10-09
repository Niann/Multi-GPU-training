#include "model.cuh"
#include "mnist.h"

#define IDX2C(i, j, ld) ((( i )*( ld ))+( j )) // ld - leading dimension

class NNModel {
private:
	vector<Layer> layers;
	int batch_size;
	int feature_size;
	int out_size;
	float learning_rate;

public:
	NNModel(int in, int out, float lr, int batch_size, vector<int> layer_size) {
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

	void train(vector<vector<float>> data,vector<int> label) {
		if (data[0].size() != this->batch_size) {
			cout << "data error! \n";
			return;
		}
		float* X = (float *)malloc(data[0].size() * this->feature_size * sizeof(float));
		float* Y = (float *)malloc(data[0].size() * this->out_size * sizeof(float));
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
	}
	
	void epoch(vector<vector<float>> &data, vector<int> &label) {
		for (int i = 0; i < data.size() + batch_size; i += batch_size) {
			vector<vector<float>> batch_data;
			vector<int> batch_label;
			for (int j = 0; j < batch_size; j++) {
				batch_data.push_back(data[i + j]);
				batch_label.push_back(label[i + j]);
			}
			this->train(batch_data, batch_label);
			cout << i << endl;
		}
	}

	float accuracy(vector<vector<float>> &data,vector<int> &label) {
		float* X = (float *)malloc(data[0].size() * this->feature_size * sizeof(float));
		for (int i = 0; i < data.size(); i++) {
			for (int j = 0; j < this->feature_size; j++) {
				X[IDX2C(i, j, this->feature_size)] = data[i][j];
			}
		}

		for (int i = 0; i < this->layers.size(); i++) {
			X = this->layers[i].forward(X, data.size());
		}
	}
};

int main() {
	string filename = "t10k-images-idx3-ubyte";
	int number_of_images = 10000;
	int image_size = 28 * 28;


	//read MNIST iamge into double vector
	vector<vector<float> > train_X;
	read_Mnist(filename, train_X);


	string filename1 = "t10k-labels-idx1-ubyte";

	//read MNIST label into double vector
	vector<float> ttrain_y(number_of_images);
	read_Mnist_Label(filename1, ttrain_y);
	vector<int> train_y(number_of_images);
	for (int i = 0; i < train_y.size(); i++) {
		train_y.push_back((int)ttrain_y[i]);
	}
	
	cout << "data loaded" << endl;
	
	vector<int> layers;
	layers.push_back(3);
	layers.push_back(3);

	NNModel* model = new NNModel(768,10,0.01,32, layers);
	model->train(train_X,train_y);

	return 0;
}