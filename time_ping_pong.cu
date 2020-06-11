#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>

#define PPN 4

double timePingPong(int cpu0, int cpu1, float* data, int size, int n_tests = 1000)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Status status;
    int ping_tag = 1234;
    int pong_tag = 4321;

    double t0, tfinal;

    // Warm Up
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        if (rank == cpu0)
        {
            MPI_Send(data, size, MPI_FLOAT, cpu1, ping_tag, MPI_COMM_WORLD);
            MPI_Recv(data, size, MPI_FLOAT, cpu1, pong_tag, MPI_COMM_WORLD, &status);
        }
        else if (rank == cpu1)
        {
            MPI_Recv(data, size, MPI_FLOAT, cpu0, ping_tag, MPI_COMM_WORLD, &status);
            MPI_Send(data, size, MPI_FLOAT, cpu0, pong_tag, MPI_COMM_WORLD);
        }
    }
    tfinal = ((MPI_Wtime() - t0) / (2 * n_tests)) * 1000;

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        if (rank == cpu0)
        {
            MPI_Send(data, size, MPI_FLOAT, cpu1, ping_tag, MPI_COMM_WORLD);
            MPI_Recv(data, size, MPI_FLOAT, cpu1, pong_tag, MPI_COMM_WORLD, &status);
        }
        else if (rank == cpu1)
        {
            MPI_Recv(data, size, MPI_FLOAT, cpu0, ping_tag, MPI_COMM_WORLD, &status);
            MPI_Send(data, size, MPI_FLOAT, cpu0, pong_tag, MPI_COMM_WORLD);
        }
    }
    tfinal = ((MPI_Wtime() - t0) / (2 * n_tests)) * 1000;

    if (rank != cpu0 && rank != cpu1) tfinal = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return t0;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    float* x;
    int max_i = 26;
    int max_n = pow(2, max_i - 1);
    int max_bytes = max_n * sizeof(float);
    double time;

    cudaMallocHost((void**)&x, max_bytes);
    for (int cpu0 = 0; cpu0 < num_procs; cpu0++)
    {
        for (int cpu1 = cpu0+1; cpu1 < num_procs; cpu1++)
        {
int n_tests = 1000;
            if (rank == 0) printf("Pinned CPU %d and CPU %d:\t", cpu0, cpu1);
            for (int i = 0; i < max_i; i++)
            {
if (i > 14) n_tests = 100;
if (i > 20) n_tests = 10;
                time = timePingPong(cpu0, cpu1, x, pow(2, i), n_tests);
                if (rank == 0) printf("%2.5f\t", time);
            }
            if (rank == 0) printf("\n");
        }
    }
    cudaFreeHost(x);

    int gpu = rank / 2;
    cudaSetDevice(gpu);
    cudaMalloc((void**)&x, max_bytes);
    for (int cpu0 = 0; cpu0 < num_procs; cpu0++)
//int cpu0 = 0;
    {
        for (int cpu1 = cpu0 + 1; cpu1 < num_procs; cpu1++)
        {
int n_tests = 1000;
            if (rank == 0) printf("GPU on %d and GPU on %d:\t", cpu0, cpu1);
            for (int i = 0; i < max_i; i++)
            {
if (i > 14) n_tests = 100;
if (i > 20) n_tests = 10;
                time = timePingPong(cpu0, cpu1, x, pow(2, i), n_tests);
                if (rank == 0) printf("%2.5f\t", time);
            }
            if (rank == 0) printf("\n");
        }
    }
    cudaFree(x);

    MPI_Finalize();
}
