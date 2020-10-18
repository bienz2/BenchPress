#!/bin/bash
#BSUB -J all_reduce_32
#BSUB -e all_reduce_32.%J.err
#BSUB -o all_reduce_32.%J.out
#BSUB -nnodes 32
#BSUB -W 00:15
#BSUB -P CSC422
#BSUB -alloc_flags "gpumps gpudefault smt1"

module load gcc
module load cmake/3.18.1
module load cuda

cd /ccs/home/bienz/BenchPress/build/examples

jsrun -a36 -c36 -g6 -r1 -n32 -M "-gpu" --latency_priority=gpu-cpu --launch_distribution=packed ./time_allreduce

