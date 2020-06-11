#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>
#include "cuda_profiler_api.h"

#define PPN 4

void profilePingPong(int cpu0, int cpu1, float* data, int size, int n_tests = 1000)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Status status;
    int ping_tag = 1234;
    int pong_tag = 4321;

    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
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
    MPI_Barrier(MPI_COMM_WORLD);
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
    int max_i = 20;
    int max_n = pow(2, max_i - 1);
    int max_bytes = max_n * sizeof(float);

/*    x = (float*)malloc(max_bytes);
    for (int cpu0 = 0; cpu0 < num_procs; cpu0++)
    {
        for (int cpu1 = cpu0+1; cpu1 < num_procs; cpu1++)
        {
            if (rank == 0) printf("CPU %d and CPU %d:\n", cpu0, cpu1);
            for (int i = 0; i < max_i; i++)
            {
                profilePingPong(cpu0, cpu1, x, pow(2, i));
            }
        }
    }
    free(x);


    cudaMallocHost((void**)&x, max_bytes);
    for (int cpu0 = 0; cpu0 < num_procs; cpu0++)
    {
        for (int cpu1 = cpu0+1; cpu1 < num_procs; cpu1++)
        {
            if (rank == 0) printf("Pinned CPU %d and CPU %d:\n", cpu0, cpu1);
            for (int i = 0; i < max_i; i++)
            {
                profilePingPong(cpu0, cpu1, x, pow(2, i));
            }
        }
    }
    cudaFreeHost(x);
*/
    int gpu = rank / 2;
    cudaSetDevice(gpu);
    cudaMalloc((void**)&x, max_bytes);
    for (int cpu0 = 0; cpu0 < num_procs; cpu0++)
    {
        for (int cpu1 = cpu0 + 1; cpu1 < num_procs; cpu1++)
        {
            if (rank == 0) printf("GPU on %d and GPU on %d:\n", cpu0, cpu1);
            for (int i = 0; i < max_i; i++)
            {
                profilePingPong(cpu0, cpu1, x, pow(2, i));
            }
        }
break;

    }
    cudaFree(x);


    MPI_Finalize();
}
