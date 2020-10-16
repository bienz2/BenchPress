#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>

#ifndef ALLTOALLV_TIMER_HPP
#define ALLTOALLV_TIMER_HPP

double time_alltoallv(int size, float* gpu_data, MPI_Comm& group_comm,
        int n_tests = 1000);
double time_alltoallv_3step(int size, float* cpu_data, float* gpu_data,
        cudaStream_t& stream, MPI_Comm& group_comm, int n_tests = 1000);
double time_alltoallv_3step_msg(int size, float* cpu_data, float* gpu_data,
       int ppg, int node_rank, cudaStream_t& stream, MPI_Comm& group_comm, 
       int n_tests = 1000);

#endif
