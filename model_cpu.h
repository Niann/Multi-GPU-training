#include "layer_cpu.h"
#include "mnist.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <mpi.h>

using namespace std;

class Model_cpu {
private:
	vector<Layer_cpu*> layers;
	int batch_size;
	int feature_size;
	int out_size;
	float learning_rate;
	float* X; // place holder for X
	float* Y; // place holder for Y

public:
	Model_cpu(int in, int out, float lr, int batch_size, vector<int> layer_size);
	void train(vector<vector<float>> data, vector<int> label);
	void epoch(vector<vector<float>> &data, vector<int> &label);
	float accuracy(vector<vector<float>> &data, vector<int> &label);
	void freeMemory();
};
