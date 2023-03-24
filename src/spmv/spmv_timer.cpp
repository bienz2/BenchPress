#include "spmv_timer.h"

void time_cuda_aware_spmv(GPUMat* d_A_on, GPUMat* d_A_off, double* d_x,
        double* d_b, double* d_x_dist, ParComm* A_comm, int* d_sendidx, 
        double* d_sendbuf, cudaStream_t& stream, MPI_Comm gpu_comm, int n_tests)
{
    if (gpu_rank == 0)
    {
        // Warm-Up
	cuda_aware_spmv(d_A_on, d_A_off, d_x, d_b, d_x_dist,
                A_comm, d_sendidx, d_sendbuf, stream, gpu_comm);

        // Synchronize and time
        cudaDeviceSynchronize();
        MPI_Barrier(gpu_comm);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_tests; i++)
        {
            cuda_aware_spmv(d_A_on, d_A_off, d_x, d_b, d_x_dist,
                    A_comm, d_sendidx, d_sendbuf, stream, gpu_comm);
        }
        tfinal = (MPI_Wtime() - t0) / n_tests;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, gpu_comm);
        if (rank == 0) printf("Cuda Aware : %e\n", t0);
    }
}

void time_copy_to_cpu_spmv(GPUMat* d_A_on, GPUMat* d_A_off, double* x, double* x_dist,
        double* d_x, double* d_b, double* d_x_dist, ParComm* A_comm, int* d_sendidx, 
        double* d_sendbuf, double* sendbuf, cudaStream_t& stream, MPI_Comm gpu_comm, int n_tests)
{
    if (gpu_rank == 0)
    {
        // Warm-Up
        copy_to_cpu_spmv(d_A_on, d_A_off, x, x_dist, d_x, d_b, d_x_dist, A->comm,
                d_sendidx, d_sendbuf, sendbuf, stream, gpu_comm);

        // Synchronize and Time
        cudaDeviceSynchronize();
        MPI_Barrier(gpu_comm);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_tests; i++)
        {
            copy_to_cpu_spmv(d_A_on, d_A_off, x, x_dist, d_x, d_b, d_x_dist, A->comm,
                    d_sendidx, d_sendbuf, sendbuf, stream, gpu_comm);
        }
        tfinal = (MPI_Wtime() - t0) / n_tests;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, gpu_comm);
        if (rank == 0) printf("Copy To CPU : %e\n", t0);
    }
}

void time_copy_nap_spmv(GPUMat* d_A_on, GPUMat* d_A_off, double* d_x, double* d_x_dist,
        double* d_b, int* d_nap_sendidx, double* d_nap_sendbuf, double* nap_sendbuf,
        int* d_nap_recvidx, double* d_nap_recvbuf, double* nap_recvbuf, TAPComm* tap_comm,
        cudaStream& stream, MPI_Comm gpu_comm, int n_tests)
{
    if (gpu_rank == 0)
    {
        // Warm-Up
        copy_nap_spmv(d_A_on, d_A_off, d_x, d_x_dist, d_b,
            d_nap_sendidx, d_nap_sendbuf, nap_sendbuf,
            d_nap_recvidx, d_nap_recvbuf, nap_recvbuf,
            tap_comm, stream, gpu_comm);

        // Synchronize and Time
        cudaDeviceSynchronize();
        MPI_Barrier(gpu_comm);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_tests; i++)
        {
            copy_nap_spmv(d_A_on, d_A_off, d_x, d_x_dist, d_b,
                d_nap_sendidx, d_nap_sendbuf, nap_sendbuf,
                d_nap_recvidx, d_nap_recvbuf, nap_recvbuf,
                tap_comm, stream, gpu_comm);
        }
        tfinal = (MPI_Wtime() - t0) / n_tests;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, gpu_comm);
        if (rank == 0) printf("Node Aware : %e\n", t0);
    }
    else
    {
        // Warm-Up
        extra_comm(tap_comm);

        // Timing
        for (int i = 0; i < n_tests; i++)
        {
            extra_comm(tap_comm);
        }
    }
}

void time_dup_nap_spmv(GPUMat* d_A_on, GPUMat* d_A_off, double* d_x, double* d_x_dist, 
        double* d_b, int* d_dup_sendidx, double* d_dup_sendbuf, double* dub_sendbuf,
        int* d_dup_recvidx, double* d_dup_recvbuf, double* dup_recvbuf, TAPComm* tap_comm,
        cudaStream_t& stream, MPI_Comm node_gpu_comm, MPI_Comm gpu_comm, int n_tests)
{
    if (gpu_rank == 0)
    {
        // Warm-Up
        dup_nap_spmv(d_A_on, d_A_off, d_x, d_x_dist, d_b,
            d_dup_sendidx, d_dup_sendbuf, dup_sendbuf,
            d_dup_recvidx, d_dup_recvbuf, dup_recvbuf,
            tap_comm, stream, node_gpu_comm);

        // Synchronize and Time
        cudaDeviceSynchronize();
        MPI_Barrier(gpu_comm);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_tests; i++)
        {
            dup_nap_spmv(d_A_on, d_A_off, d_x, d_x_dist, d_b,
                d_dup_sendidx, d_dup_sendbuf, dup_sendbuf,
                d_dup_recvidx, d_dup_recvbuf, dup_recvbuf,
                tap_comm, stream, node_gpu_comm);
        }
        tfinal = (MPI_Wtime() - t0) / n_tests;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, gpu_comm);
        if (rank == 0) printf("Dup Devptr Node Aware : %e\n", t0);
    }
    else
    {
        // Warm-Up
        dup_nap_spmv(NULL, NULL, d_x, d_x_dist, NULL,
            d_dup_sendidx, d_dup_sendbuf, dup_sendbuf,
            d_dup_recvidx, d_dup_recvbuf, dup_recvbuf,
            tap_comm, stream, node_gpu_comm);

        // Timing
        for (int i = 0; i < n_tests; i++)
        {
            dup_nap_spmv(NULL, NULL, d_x, d_x_dist, NULL,
                d_dup_sendidx, d_dup_sendbuf, dup_sendbuf,
                d_dup_recvidx, d_dup_recvbuf, dup_recvbuf,
                tap_comm, stream, node_gpu_comm);
        }
    }
}
