#include "model.h"
#include "mnist.h"
#include <time.h>

#define INPUT_SIZE 784
#define LABEL_SIZE 10
#define BATCH_SIZE 64
#define EPOCH 10


int main(int argc, char **argv) {
	/*
	int comm_size;
	int rank;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	*/
	int rank = 0;
	int comm_size = 1;

	string Xtrain_file = "train-images-idx3-ubyte";
	string Ytrain_file = "train-labels-idx1-ubyte";
	string Xtest_file = "t10k-images-idx3-ubyte";
	string Ytest_file = "t10k-labels-idx1-ubyte";

	//read MNIST iamge into float vector
	vector<vector<float>> train_X;
	vector<vector<float>> test_X;
	read_Mnist(Xtrain_file, train_X, rank, comm_size);
	read_Mnist(Xtest_file, test_X, rank, comm_size);
	cout << "training set size: " << train_X.size() << endl;
	cout << "test set size: " << test_X.size() << endl;

	//read MNIST label into int vector
	vector<int> train_y;
	vector<int> test_y;
	read_Mnist_Label(Ytrain_file, train_y, rank, comm_size);
	read_Mnist_Label(Ytest_file, test_y, rank, comm_size);
	cout << train_y.size() << endl;
	cout << test_y.size() << endl;
	cout << "data loaded" << endl;

	vector<int> layers;
	layers.push_back(300);

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