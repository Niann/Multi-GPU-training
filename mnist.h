#pragma once

#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

void read_Mnist(string filename, vector<vector<double> > &vec);

void read_Mnist_Label(string filename, vector<double> &vec);