#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>

#include "utils/utils.hpp"

#ifndef ALLTOALLV_TIMER_HPP
#define ALLTOALLV_TIMER_HPP

double time_alltoallv(int size, float* gpu_send_data, float* gpu_recv_data,
        MPI_Comm& group_comm, int n_tests = 1000);
double time_alltoallv_3step(int size, float* cpu_send_data, float* cpu_recv_data,
        float* gpu_data, gpuStream_t& stream, MPI_Comm& group_comm, int n_tests = 1000);
double time_alltoallv_3step_msg(int size, float* cpu_send_data, float* cpu_recv_data,
       float* gpu_data, int ppg, int node_rank, gpuStream_t& stream, MPI_Comm& group_comm, 
       int n_tests = 1000);
double time_alltoallv_3step_dup(int size, float* cpu_send_data, float* cpu_recv_data,
       float* gpu_data, int ppg, int node_rank, gpuStream_t& stream, MPI_Comm& group_comm,
       int n_tests);


double time_alltoallv_imsg(int size, float* gpu_send_data, float* gpu_recv_data,
        MPI_Comm& group_comm, int n_tests = 1000);
double time_alltoallv_3step_imsg(int size, float* cpu_send_data, float* cpu_recv_data,
        float* gpu_data, gpuStream_t& stream, MPI_Comm& group_comm, int n_tests = 1000);
double time_alltoallv_3step_msg_imsg(int size, float* cpu_send_data, float* cpu_recv_data,
       float* gpu_data, int ppg, int node_rank, gpuStream_t& stream, MPI_Comm& group_comm, 
       int n_tests = 1000);
double time_alltoallv_3step_dup_imsg(int size, float* cpu_send_data, float* cpu_recv_data,
       float* gpu_data, int ppg, int node_rank, gpuStream_t& stream, MPI_Comm& group_comm,
       int n_tests);

#endif
