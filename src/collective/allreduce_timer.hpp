#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>
#include "utils/utils.hpp"

#ifndef ALLREDUCE_TIMER_HPP
#define ALLREDUCE_TIMER_HPP

double time_allreduce(int size, float* gpu_data, MPI_Comm& group_comm,
        int n_tests = 1000);
double time_allreduce_3step(int size, float* cpu_data, float* gpu_data,
        gpuStream_t& stream, MPI_Comm& group_comm, int n_tests = 1000);
double time_allreduce_3step_msg(int size, float* cpu_data, float* gpu_data,
       int ppg, int node_rank, gpuStream_t& stream, MPI_Comm& group_comm, 
       int n_tests = 1000);

#endif
