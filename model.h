#include "layer.cuh"
#include "mnist.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <mpi.h>

using namespace std;

class Model {
private:
	vector<Layer*> layers;
	int batch_size;
	int feature_size;
	int out_size;
	float learning_rate;
	float* X; // place holder for X
	float* Y; // place holder for Y
	float* d_X; // X on device
	float* d_Y; // Y on device

public:
	Model(int in, int out, float lr, int batch_size, vector<int> layer_size, int gpu);
	void train(vector<vector<float>> data, vector<int> label);
	void epoch(vector<vector<float>> &data, vector<int> &label);
	float accuracy(vector<vector<float>> &data, vector<int> &label);
	void freeMemory();
};