require MNIST datasets.

To compile **GPU** version:  

module load OpenMPI/3.0.0-GCC-6.2.0-cuda9-ucx CUDA/9.0.176-GCC-6.2.0  

nvcc -ccbin=mpic++ -lcublas -o gpu layer.cu mnist.cpp model.cpp

To compile **CPU** version:

module load OpenMPI/3.0.0 GCC/6.2.0

mpic++ -o cpu -fopenmp -std=c++11 layer_cpu.cpp mnist.cpp model_cpu.cpp
