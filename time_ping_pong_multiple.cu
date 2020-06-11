#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>

#define PPN 4

double timePingPong(int cpu0, int cpu1, float* data, int size, int n_msgs, 
    MPI_Request* requests, int n_tests = 1000)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int ping_tag = 12345;
    int pong_tag = 54321;

    double t0, tfinal;

    size = size / n_msgs;

    // Warm Up
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        if (rank == cpu0)
        {
            for (int i = 0; i < n_msgs; i++)
            {
                MPI_Isend(data, size, MPI_FLOAT, cpu1, ping_tag + i, MPI_COMM_WORLD, &requests[i]);
            }
            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
            for (int i = 0; i < n_msgs; i++)
            {
                MPI_Irecv(data, size, MPI_FLOAT, cpu1, pong_tag + i, MPI_COMM_WORLD, &requests[i]);
            }
            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
        }
        else if (rank == cpu1)
        {
            for (int i = 0; i < n_msgs; i++)
            {
                MPI_Irecv(data, size, MPI_FLOAT, cpu0, ping_tag + i, MPI_COMM_WORLD, &requests[i]);
            }
            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
            for (int i = 0; i < n_msgs; i++)
            {
                MPI_Isend(data, size, MPI_FLOAT, cpu0, pong_tag + i, MPI_COMM_WORLD, &requests[i]);
            }
            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
        }
    }
    tfinal = ((MPI_Wtime() - t0) / (2 * n_tests)) * 1000;

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        if (rank == cpu0)
        {
            for (int i = 0; i < n_msgs; i++)
            {
                MPI_Isend(data, size, MPI_FLOAT, cpu1, ping_tag + i, MPI_COMM_WORLD, &requests[i]);
            }
            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
            for (int i = 0; i < n_msgs; i++)
            {
                MPI_Irecv(data, size, MPI_FLOAT, cpu1, pong_tag + i, MPI_COMM_WORLD, &requests[i]);
            }
            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
        }
        else if (rank == cpu1)
        {
            for (int i = 0; i < n_msgs; i++)
            {
                MPI_Irecv(data, size, MPI_FLOAT, cpu0, ping_tag + i, MPI_COMM_WORLD, &requests[i]);
            }
            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
            for (int i = 0; i < n_msgs; i++)
            {
                MPI_Isend(data, size, MPI_FLOAT, cpu0, pong_tag + i, MPI_COMM_WORLD, &requests[i]);
            }
            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
        }
    }
    tfinal = ((MPI_Wtime() - t0) / (2 * n_tests)) * 1000;

    if (rank != cpu0 && rank != cpu1) tfinal = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return t0;
}

double timePingPongQueue(int cpu0, int cpu1, float* data, int size, int n_msgs, 
    MPI_Request* requests, int n_tests = 1000)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int send_ping_tag = 12345;
    int send_pong_tag = 54321;
    int recv_ping_tag = send_ping_tag + n_msgs - 1;
    int recv_pong_tag = send_pong_tag + n_msgs - 1;

    double t0, tfinal;

    size = size / n_msgs;

    // Warm Up
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        if (rank == cpu0)
        {
            for (int i = 0; i < n_msgs; i++)
            {
                MPI_Isend(data, size, MPI_FLOAT, cpu1, send_ping_tag + i, MPI_COMM_WORLD, &requests[i]);
            }
            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
            for (int i = 0; i < n_msgs; i++)
            {
                MPI_Irecv(data, size, MPI_FLOAT, cpu1, recv_pong_tag - i, MPI_COMM_WORLD, &requests[i]);
            }
            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
        }
        else if (rank == cpu1)
        {
            for (int i = 0; i < n_msgs; i++)
            {
                MPI_Irecv(data, size, MPI_FLOAT, cpu0, recv_ping_tag - i, MPI_COMM_WORLD, &requests[i]);
            }
            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
            for (int i = 0; i < n_msgs; i++)
            {
                MPI_Isend(data, size, MPI_FLOAT, cpu0, send_pong_tag + i, MPI_COMM_WORLD, &requests[i]);
            }
            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
        }
    }
    tfinal = ((MPI_Wtime() - t0) / (2 * n_tests)) * 1000;

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        if (rank == cpu0)
        {
            for (int i = 0; i < n_msgs; i++)
            {
                MPI_Isend(data, size, MPI_FLOAT, cpu1, send_ping_tag + i, MPI_COMM_WORLD, &requests[i]);
            }
            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
            for (int i = 0; i < n_msgs; i++)
            {
                MPI_Irecv(data, size, MPI_FLOAT, cpu1, recv_pong_tag - i, MPI_COMM_WORLD, &requests[i]);
            }
            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
        }
        else if (rank == cpu1)
        {
            for (int i = 0; i < n_msgs; i++)
            {
                MPI_Irecv(data, size, MPI_FLOAT, cpu0, recv_ping_tag - i, MPI_COMM_WORLD, &requests[i]);
            }
            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
            for (int i = 0; i < n_msgs; i++)
            {
                MPI_Isend(data, size, MPI_FLOAT, cpu0, send_pong_tag + i, MPI_COMM_WORLD, &requests[i]);
            }
            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
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
    int max_i = 14;
    int max_s = 14;
    int max_n = pow(2, max_i - 1);
    int max_n_msgs = pow(2, max_s - 1);
    int max_bytes = max_n * sizeof(float);
    double time;
    std::vector<MPI_Request> requests(max_n_msgs);
    int n_tests;

    int gpu = rank / 2;
    cudaSetDevice(gpu);
    cudaMalloc((void**)&x, max_bytes);

    int cpu_list[3] = {1,2,4};

    int cpu0 = 0;
    {
        for (int k = 0; k < 3; k++)
        {
            int cpu1 = cpu_list[k];;
            if (rank == 0) printf("GPU on %d and GPU on %d:\n", cpu0, cpu1);
            n_tests = 1000;
            for (int i = 0; i < max_i; i++)
            {
                int bytes = pow(2, i);
                if (rank == 0) printf("%d:\t", bytes);
                for (int s = 0; s <= i; s++)
                {
                    if (s > 6) n_tests = 100;
                    time = timePingPong(cpu0, cpu1, x, bytes, pow(2, s), requests.data(), n_tests);
                    if (rank == 0) printf("%2.5f\t", time);
                }
                if (rank == 0) printf("\n");
            }
            if (rank == 0) printf("\n");


            if (rank == 0) printf("QUEUE: GPU on %d and GPU on %d:\n", cpu0, cpu1);
            n_tests = 1000;
            for (int i = 0; i < max_i; i++)
            {
                int bytes = pow(2, i);
                if (rank == 0) printf("%d:\t", bytes);
                for (int s = 0; s <= i; s++)
                {
                    if (s > 6) n_tests = 100;
                    time = timePingPongQueue(cpu0, cpu1, x, bytes, pow(2, s), requests.data(), n_tests);
                    if (rank == 0) printf("%2.5f\t", time);
                }
                if (rank == 0) printf("\n");
            }
            if (rank == 0) printf("\n");
        }
    }
    cudaFree(x);

    MPI_Finalize();
}
