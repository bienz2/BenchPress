#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>

#define PPN 4

float timePingPong(int local_rank, int n, float* data, int size, bool active, int n_tests = 1000)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (n > 10000) n_tests = 1000;

    MPI_Status status;
    int ping_tag = 1234;
    int pong_tag = 4321;
    bool first = false;
    bool second = false;

    float t0, tfinal;

    int proc;
    if (local_rank < n && active)
    {
        if (rank % 2 == 0)
        {
            first = true;
            proc = rank + 1;
        }
        else 
        {
            second = true;
            proc = rank - 1;
        }
    }

    // Warm Up
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        if (first)
        {
            MPI_Send(data, size, MPI_FLOAT, proc, ping_tag, MPI_COMM_WORLD);
            MPI_Recv(data, size, MPI_FLOAT, proc, pong_tag, MPI_COMM_WORLD, &status);
        }
        else if (second)
        {
            MPI_Recv(data, size, MPI_FLOAT, proc, ping_tag, MPI_COMM_WORLD, &status);
            MPI_Send(data, size, MPI_FLOAT, proc, pong_tag, MPI_COMM_WORLD);
        }
    }
    tfinal = (MPI_Wtime() - t0) / (2 * n_tests) * 1000;


    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        if (first)
        {
            MPI_Send(data, size, MPI_FLOAT, proc, ping_tag, MPI_COMM_WORLD);
            MPI_Recv(data, size, MPI_FLOAT, proc, pong_tag, MPI_COMM_WORLD, &status);
        }
        else if (second)
        {
            MPI_Recv(data, size, MPI_FLOAT, proc, ping_tag, MPI_COMM_WORLD, &status);
            MPI_Send(data, size, MPI_FLOAT, proc, pong_tag, MPI_COMM_WORLD);
        }
    }
    tfinal = (MPI_Wtime() - t0) / (2 * n_tests) * 1000;
    if (!first && !second) tfinal = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&tfinal, &t0, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
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
    int max_i = 20;
    int max_n = pow(2, max_i - 1);
    int max_bytes = max_n * sizeof(float);
    float time;

    int procs_per_socket = 18;
    int procs_per_node = procs_per_socket * 2;
    int gpus_per_socket = 2;
    int gpus_per_node = gpus_per_socket*2;
    
    int node_rank = rank / 2;
    int socket_rank = node_rank % procs_per_socket;
printf("Rank %d, Socket Rank %d, Node Rank %d\n", rank, socket_rank, node_rank);
    int socket_idx = node_rank / procs_per_socket;
    int gpu = num_gpus+1;
    if (socket_rank < gpus_per_socket)
        gpu = socket_idx * 2 + socket_rank;


/*    if (rank == 0) printf("CPU Ping Pong\n");
    x = (float*)malloc(max_bytes);
    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2,i);
        if (rank == 0) printf("%d\t", s);
        for (int n = 0; n < procs_per_node; n++)
        {
            time = timePingPong(node_rank, n+1, x, s, true);
            if (rank == 0) printf("%2.5f\t", time);
        }
        if (rank == 0) printf("\n");
    }

    if (rank == 0) printf("CPU PerSocket Ping Pong\n");
    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2,i);
        if (rank == 0) printf("%d\t", s);
        for (int n = 0; n < procs_per_socket; n++)
        {
            time = timePingPong(socket_rank, n+1, x, s, true);
            if (rank == 0) printf("%2.5f\t", time);
        }
        if (rank == 0) printf("\n");
    }
    free(x);
*/

/*
    if (gpu < num_gpus)
    {
        cudaSetDevice(gpu);
        cudaMalloc((void**)&x, max_bytes);
    }
    if (rank == 0) printf("GPU Ping Pong\n");
    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2,i);
        if (rank == 0) printf("%d\t", s);
        for (int n = 0; n < num_gpus; n++)
        {
            time = timePingPong(socket_rank, n+1, x, s, gpu < num_gpus, 1000);
            if (rank == 0) printf("%2.5f\t", time);
        }
        if (rank == 0) printf("\n");
    }

    if (rank == 0) printf("GPU 1PerSocket Ping Pong\n");
    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2,i);
        if (rank == 0) printf("%d\t", s);
        for (int n = 0; n < 2; n++)
        {
            time = timePingPong(socket_idx, n+1, x, s, socket_rank == 0, 1000);
            if (rank == 0) printf("%2.5f\t", time);
        }
        if (rank == 0) printf("\n");
    }
    if (gpu < num_gpus)
        cudaFree(x);
*/

    MPI_Finalize();
}
