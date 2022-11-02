#ifndef SPMV_TIMER_H
#define SPMV_TIMER_H

#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>

void time_cuda_aware_spmv(GPUMat* d_A_on, GPUMat* d_A_off, double* d_x,
        double* d_b, double* d_x_dist, ParComm* A_comm, int* d_sendidx, 
        double* d_sendbuf, cudaStream_t& stream, MPI_Comm gpu_comm, int n_tests);
void time_copy_to_cpu_spmv(GPUMat* d_A_on, GPUMat* d_A_off, double* x, double* x_dist,
        double* d_x, double* d_b, double* d_x_dist, ParComm* A_comm, int* d_sendidx, 
        double* d_sendbuf, double* sendbuf, cudaStream_t& stream, MPI_Comm gpu_comm, int n_tests);
void time_copy_nap_spmv(GPUMat* d_A_on, GPUMat* d_A_off, double* d_x, double* d_x_dist,
        double* d_b, int* d_nap_sendidx, double* d_nap_sendbuf, double* nap_sendbuf,
        int* d_nap_recvidx, double* d_nap_recvbuf, double* nap_recvbuf, TAPComm* tap_comm,
        cudaStream& stream, MPI_Comm gpu_comm, int n_tests);
void time_dup_nap_spmv(GPUMat* d_A_on, GPUMat* d_A_off, double* d_x, double* d_x_dist, 
        double* d_b, int* d_dup_sendidx, double* d_dup_sendbuf, double* dub_sendbuf,
        int* d_dup_recvidx, double* d_dup_recvbuf, double* dup_recvbuf, TAPComm* tap_comm,
        cudaStream_t& stream, MPI_Comm node_gpu_comm, MPI_Comm gpu_comm, int n_tests);

#endif
