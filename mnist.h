#pragma once

#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

void read_Mnist(string filename, vector<vector<float>> &vec, int rank, int comm_size);

void read_Mnist_Label(string filename, vector<int> &vec, int rank, int comm_size);