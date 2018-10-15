To compile:  

module load OpenMPI/3.0.0-GCC-6.2.0-cuda9-ucx CUDA/9.0.176-GCC-6.2.0  

nvcc -ccbin=mpic++ -lcublas -o test layer.cu mnist.cpp model.cpp
