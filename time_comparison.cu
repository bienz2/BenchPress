#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>

#define PPN 4

double timeCudaAware(int cpu0, int cpu1, float* data, int size, int n_msgs, 
    MPI_Request* requests, int n_tests = 1000)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int ping_tag = 12345;
    int pong_tag = 54321;

    double t0, tfinal;

    size = size / n_msgs;

    // Warm Up
    cudaDeviceSynchronize();
    cudaStreamSynchronize(0);
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        if (rank == cpu0)
        {
            for (int i = 0; i < n_msgs; i++)
            {
                MPI_Isend(&(data[i*size]), size, MPI_FLOAT, cpu1, ping_tag + i, MPI_COMM_WORLD, &requests[i]);
            }
            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
            for (int i = 0; i < n_msgs; i++)
            {
                MPI_Irecv(&(data[i*size]), size, MPI_FLOAT, cpu1, pong_tag + i, MPI_COMM_WORLD, &requests[i]);
            }
            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
        }
        else if (rank == cpu1)
        {
            for (int i = 0; i < n_msgs; i++)
            {
                MPI_Irecv(&(data[i*size]), size, MPI_FLOAT, cpu0, ping_tag + i, MPI_COMM_WORLD, &requests[i]);
            }
            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
            for (int i = 0; i < n_msgs; i++)
            {
                MPI_Isend(&(data[i*size]), size, MPI_FLOAT, cpu0, pong_tag + i, MPI_COMM_WORLD, &requests[i]);
            }
            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
        }
    }
    tfinal = ((MPI_Wtime() - t0) / (2 * n_tests)) * 1000;

    cudaDeviceSynchronize();
    cudaStreamSynchronize(0);
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        if (rank == cpu0)
        {
            for (int i = 0; i < n_msgs; i++)
            {
                MPI_Isend(&(data[i*size]), size, MPI_FLOAT, cpu1, ping_tag + i, MPI_COMM_WORLD, &requests[i]);
            }
            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
            for (int i = 0; i < n_msgs; i++)
            {
                MPI_Irecv(&(data[i*size]), size, MPI_FLOAT, cpu1, pong_tag + i, MPI_COMM_WORLD, &requests[i]);
            }
            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
        }
        else if (rank == cpu1)
        {
            for (int i = 0; i < n_msgs; i++)
            {
                MPI_Irecv(&(data[i*size]), size, MPI_FLOAT, cpu0, ping_tag + i, MPI_COMM_WORLD, &requests[i]);
            }
            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
            for (int i = 0; i < n_msgs; i++)
            {
                MPI_Isend(&(data[i*size]), size, MPI_FLOAT, cpu0, pong_tag + i, MPI_COMM_WORLD, &requests[i]);
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


double timeThreeStep(int cpu0, int cpu1, float* data, int size, int n_msgs, 
    MPI_Request* requests, int n_tests = 1000)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int ping_tag = 12345;
    int pong_tag = 54321;

    std::vector<float> cpu_data(size);

    double t0, tfinal;

//    int bytes = size * sizeof(float);
    size = size / n_msgs;
    int bytes = size * sizeof(float);

    // Warm Up
    cudaDeviceSynchronize();
    cudaStreamSynchronize(0);
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        if (rank == cpu0)
        {
            cudaMemcpyAsync(cpu_data.data(), data, bytes, cudaMemcpyDeviceToHost, 0);
            cudaStreamSynchronize(0);
            for (int i = 0; i < n_msgs; i++)
            {
//                MPI_Isend(&(cpu_data[i*size]), size, MPI_FLOAT, cpu1, ping_tag + i, MPI_COMM_WORLD, &requests[i]);
MPI_Send(cpu_data.data(), size, MPI_FLOAT, cpu1, ping_tag + 1, MPI_COMM_WORLD);
            }
//            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
            for (int i = 0; i < n_msgs; i++)
            {
//                MPI_Irecv(&(cpu_data[i*size]), size, MPI_FLOAT, cpu1, pong_tag + i, MPI_COMM_WORLD, &requests[i]);
MPI_Recv(cpu_data.data(), size, MPI_FLOAT, cpu1, pong_tag+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
//            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
            cudaMemcpyAsync(data, cpu_data.data(), bytes, cudaMemcpyHostToDevice, 0);
            cudaStreamSynchronize(0);
        }
        else if (rank == cpu1)
        {
            for (int i = 0; i < n_msgs; i++)
            {
//                MPI_Irecv(&(cpu_data[i*size]), size, MPI_FLOAT, cpu0, ping_tag + i, MPI_COMM_WORLD, &requests[i]);
MPI_Recv(cpu_data.data(), size, MPI_FLOAT, cpu0, ping_tag+i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
//            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
            cudaMemcpyAsync(data, cpu_data.data(), bytes, cudaMemcpyHostToDevice, 0);
            cudaStreamSynchronize(0);
            cudaMemcpyAsync(cpu_data.data(), data, bytes, cudaMemcpyDeviceToHost, 0);
            cudaStreamSynchronize(0);
            for (int i = 0; i < n_msgs; i++)
            {
//                MPI_Isend(&(cpu_data[i*size]), size, MPI_FLOAT, cpu0, pong_tag + i, MPI_COMM_WORLD, &requests[i]);
MPI_Send(cpu_data.data(), size, MPI_FLOAT, cpu0, pong_tag+i, MPI_COMM_WORLD);
            }
//            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
        }
    }
    tfinal = ((MPI_Wtime() - t0) / (2 * n_tests)) * 1000;

    cudaDeviceSynchronize();
    cudaStreamSynchronize(0);
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        if (rank == cpu0)
        {
            cudaMemcpyAsync(cpu_data.data(), data, bytes, cudaMemcpyDeviceToHost, 0);
            cudaStreamSynchronize(0);
            for (int i = 0; i < n_msgs; i++)
            {
//                MPI_Isend(&(cpu_data[i*size]), size, MPI_FLOAT, cpu1, ping_tag + i, MPI_COMM_WORLD, &requests[i]);
MPI_Send(cpu_data.data(), size, MPI_FLOAT, cpu1, ping_tag+i, MPI_COMM_WORLD);
            }
//            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
            for (int i = 0; i < n_msgs; i++)
            {
//                MPI_Irecv(&(cpu_data[i*size]), size, MPI_FLOAT, cpu1, pong_tag + i, MPI_COMM_WORLD, &requests[i]);
MPI_Recv(cpu_data.data(), size, MPI_FLOAT, cpu1, pong_tag+i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
//            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
            cudaMemcpyAsync(data, cpu_data.data(), bytes, cudaMemcpyHostToDevice, 0);
            cudaStreamSynchronize(0);
        }
        else if (rank == cpu1)
        {
            for (int i = 0; i < n_msgs; i++)
            {
//                MPI_Irecv(&(cpu_data[i*size]), size, MPI_FLOAT, cpu0, ping_tag + i, MPI_COMM_WORLD, &requests[i]);
MPI_Recv(cpu_data.data(), size, MPI_FLOAT, cpu0, ping_tag+i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
//            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
            cudaMemcpyAsync(data, cpu_data.data(), bytes, cudaMemcpyHostToDevice, 0);
            cudaStreamSynchronize(0);
            cudaMemcpyAsync(cpu_data.data(), data, bytes, cudaMemcpyDeviceToHost, 0);
            cudaStreamSynchronize(0);
            for (int i = 0; i < n_msgs; i++)
            {
//                MPI_Isend(&(cpu_data[i*size]), size, MPI_FLOAT, cpu0, pong_tag + i, MPI_COMM_WORLD, &requests[i]);
MPI_Send(cpu_data.data(), size, MPI_FLOAT, cpu0, pong_tag+i, MPI_COMM_WORLD);
            }
//            MPI_Waitall(n_msgs, requests, MPI_STATUSES_IGNORE);
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
    int max_i = 18;
    int max_s = 18;
    int max_n = pow(2, max_i - 1);
    int max_n_msgs = pow(2, max_s - 1);
    int max_bytes = max_n * sizeof(float);
    double time;
    std::vector<MPI_Request> requests(max_n_msgs);
    int n_tests;

    int gpu = rank / 2;
    cudaSetDevice(gpu);
    cudaMalloc((void**)&x, max_bytes);

    //int cpu_list[3] = {1,2,4};

    int cpu0 = 0;
    {
      //  for (int k = 0; k < 3; k++)
        {
        //    int cpu1 = cpu_list[k];
int cpu1 = 1;

            if (rank == 0) printf("Cuda-Aware: GPU on %d and GPU on %d:\n", cpu0, cpu1);
            n_tests = 1000;
            for (int i = 0; i < max_i; i++)
            {
                int bytes = pow(2, i);
                if (rank == 0) printf("%d:\t", bytes);
                for (int s = 0; s <= i; s++)
                {
                    if (s > 6) n_tests = 100;
                    if (s > 13) n_tests = 10;
                    time = timeCudaAware(cpu0, cpu1, x, bytes, pow(2, s), requests.data(), n_tests);
                    if (rank == 0) printf("%2.5f\t", time);
                }
                if (rank == 0) printf("\n");
            }
            if (rank == 0) printf("\n");


            if (rank == 0) printf("Three Step: GPU on %d and GPU on %d:\n", cpu0, cpu1);
            n_tests = 1000;
            for (int i = 0; i < max_i; i++)
            {
                int bytes = pow(2, i);
                if (rank == 0) printf("%d:\t", bytes);
                for (int s = 0; s <= i; s++)
                {
                    if (s > 6) n_tests = 100;
                    if (s > 13) n_tests = 10;
                    time = timeThreeStep(cpu0, cpu1, x, bytes, pow(2, s), requests.data(), n_tests);
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
