#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>

float timeMemcpy(int bytes, float* orig_x, float* dest_x,
        cudaMemcpyKind copy_kind, int n_tests = 1000)
{
    float time;
    cudaEvent_t startEvent, stopEvent;

    // Warm Up
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaDeviceSynchronize();
    cudaStreamSynchronize(0);
    cudaEventRecord(startEvent, 0);
    for (int i = 0; i < n_tests; i++)
    {
        cudaMemcpyAsync(dest_x, orig_x, bytes, copy_kind, 0);
        cudaStreamSynchronize(0);
    }
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    // Time Memcpy
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaDeviceSynchronize();
    cudaStreamSynchronize(0);
    cudaEventRecord(startEvent, 0);
    for (int i = 0; i < n_tests; i++)
    {
        cudaMemcpyAsync(dest_x, orig_x, bytes, copy_kind, 0);
        cudaStreamSynchronize(0);
    }
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return time / n_tests;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    int max_i = 20;
    int max_bytes = pow(2,max_i-1) * sizeof(float);
    int bytes, size;
    float* cpu_data;
    float* gpu_data;
    cudaMallocHost((void**)&cpu_data, max_bytes);

    int procs_per_socket = num_procs / 2;
    int node_rank = rank / 2;
    int socket_rank = node_rank % procs_per_socket;
    int gpu = socket_rank % 2 + node_rank * 2;
    float t0, tfinal;
    int n_tests;

    cudaSetDevice(gpu);
    cudaMalloc((void**)&gpu_data, max_bytes);

    if (rank == 0) printf("HostToDevice:\n");
    for (int i = 0; i < max_i; i++)
    {
        size = pow(2, i);
        bytes = size * sizeof(float);
        n_tests = 1000;
        if (rank == 0) printf("%d:\t", size);
        for (int np = 1; np <= num_procs; np++)
        {
            if (np > 4) n_tests = 100;
            if (rank < np) tfinal = timeMemcpy(bytes, cpu_data, gpu_data, cudaMemcpyHostToDevice, n_tests);
            else tfinal = 0.0;
            MPI_Reduce(&tfinal, &t0, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("%e\t", t0);
        }
        if (rank == 0) printf("\n");
    }

    if (rank == 0) printf("DeviceToHost:\n");
    for (int i = 0; i < max_i; i++)
    {
        size = pow(2, i);
        bytes = size * sizeof(float);
        n_tests = 1000;
        if (rank == 0) printf("%d:\t", size);
        for (int np = 1; np <= num_procs; np++)
        {
            if (np > 4) n_tests = 100;
            if (rank < np) tfinal = timeMemcpy(bytes, gpu_data, cpu_data, cudaMemcpyDeviceToHost, n_tests);
            else tfinal = 0.0;
            MPI_Reduce(&tfinal, &t0, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("%e\t", t0);
        }
        if (rank == 0) printf("\n");
    }

    cudaFree(gpu_data);
    cudaFreeHost(cpu_data);

    MPI_Finalize();
}
