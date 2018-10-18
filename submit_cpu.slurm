#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-00:30:00
#SBATCH --partition=physical

module load OpenMPI/3.0.0

mpirun -n 4 ./cpu
