#!/bin/bash
#BSUB -J ML_Geer_8
#BSUB -e ML_Geer_8.%J.err
#BSUB -o ML_Geer_8.%J.out
#BSUB -nnodes 8
#BSUB -W 00:15
#BSUB -P CSC422
#BSUB -alloc_flags "gpumps gpudefault smt1"

module load gcc
module load cmake/3.18.1
module load cuda

cd /ccs/home/bienz/sparse_mat

echo "jsrun -a36 -c36 -g6 -r1 -n8 -M "-gpu" --latency_priority=gpu-cpu --launch_distribution=packed ./mpi_sparse_mat suitesparse/ML_Geer.pm"


jsrun -a36 -c36 -g6 -r1 -n8 -M "-gpu" --latency_priority=gpu-cpu --launch_distribution=packed  ./mpi_sparse_mat suitesparse/ML_Geer.pm




