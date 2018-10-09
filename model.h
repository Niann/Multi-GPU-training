#include "layer.cuh"
#include "mnist.h"
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class NNModel {
private:
	vector<Layer> layers;
	int batch_size;
	int feature_size;
	int out_size;
	float learning_rate;

public:
	NNModel(int in, int out, float lr, int batch_size, vector<int> layer_size);
	void train(vector<vector<float>> data, vector<int> label);
	void epoch(vector<vector<float>> &data, vector<int> &label);
	float accuracy(vector<vector<float>> &data, vector<int> &label);

};