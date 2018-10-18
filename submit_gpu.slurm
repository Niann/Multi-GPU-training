#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

module load OpenMPI/3.0.0-GCC-6.2.0-cuda9-ucx CUDA/9.0.176-GCC-6.2.0

mpirun -n 4 ./gpu
