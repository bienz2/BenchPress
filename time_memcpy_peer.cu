#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>

float timeMemcpyPeer(int bytes, float* orig_x, int orig_gpu,
        float* dest_x, int dest_gpu, int n_tests = 1000)
{
    float time;
    cudaEvent_t startEvent, stopEvent;

    // Warm Up
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaDeviceSynchronize();
    cudaEventRecord(startEvent, 0);
    for (int i = 0; i < n_tests; i++)
    {
        cudaMemcpyPeerAsync(dest_x, dest_gpu, orig_x, orig_gpu, bytes, 0);
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
    cudaEventRecord(startEvent, 0);
    for (int i = 0; i < n_tests; i++)
    {
        cudaMemcpyPeerAsync(dest_x, dest_gpu, orig_x, orig_gpu, bytes, 0);
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

    int max_i = 30;
    int max_bytes = pow(2,max_i-1) * sizeof(float);
    int n_times = 2 * max_i * num_gpus;
    int timectr, bytes;
    std::vector<float> times(n_times);
    std::vector<float> max_times(n_times);
    float* gpu0_data;
    float* gpu1_data;
    int n_tests;

    for (int proc = 0; proc < num_procs; proc++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == proc)
        {
            timectr = 0;
            for (int gpu0 = 0; gpu0 < num_gpus; gpu0++)
            {
                cudaSetDevice(gpu0);
                cudaMalloc((void**)&gpu0_data, max_bytes);
                for (int gpu1 = gpu0 + 1; gpu1 < num_gpus; gpu1++)
                {
                    cudaSetDevice(gpu1);
                    cudaMalloc((void**)&gpu1_data, max_bytes);

                    n_tests = 1000;
                    for (int i = 0; i < max_i; i++)
                    {
                        if (i > 20) n_tests = 100;
                        if (i > 25) n_tests = 10;
                        bytes = pow(2, i) * sizeof(float);
                        times[timectr++] = timeMemcpyPeer(bytes, gpu0_data, gpu0, gpu1_data, gpu1, n_tests);
                    }
                    cudaFree(gpu1_data);
                }
                cudaFree(gpu0_data);
            }
        }
        else std::fill(times.begin(), times.end(), 0);

        MPI_Reduce(times.data(), max_times.data(), times.size(), MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            timectr = 0;
            for (int gpu0 = 0; gpu0 < num_gpus; gpu0++)
            {
                for (int gpu1 = gpu0 + 1; gpu1 < num_gpus; gpu1++)
                {
                    printf("CPU %d, GPU %d to GPU %d:\t", proc, gpu0, gpu1);
                    for (int i = 0; i < max_i; i++)
                        printf("%2.5f\t", max_times[timectr++]);
                    printf("\n");
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
}
