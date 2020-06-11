#include "mpi.h"
#include "cuda_runtime.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>


__global__ void add(float* a, float* b, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) a[tid] += b[tid];
}


void cpuadd(float* a, float* b, int n)
{
    for (int i = 0; i < n; i++)
        a[i] += b[i];
}


float timeGPUSum(int bytes, float* orig, float* addl, int n_tests = 1000)
{
    float time;
    cudaEvent_t startEvent, stopEvent;

    int blockSize = 1024;
    int gridSize = (int)ceil((float)n/blockSize);
 
    // Warm Up
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaDeviceSynchronize();
    cudaEventRecord(startEvent, 0);
    for (int i = 0; i < n_tests; i++)
    {
        add<<<gridSize, blockSize>>>(orig, addl, n);
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
        add<<<gridSize, blockSize>>>(orig, addl, n);
    }
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return time / n_tests;
}


float timeCPUSum(int bytes, float* orig, float* addl, int n_tests = 1000)
{
    float t0, tfinal;

    //Warm Up
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
        cpuadd(orig, addl, 
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
    int n_times = 2 * max_i * num_gpus;
    int timectr, bytes;
    std::vector<float> times(n_times);
    std::vector<float> max_times(n_times);
    float* cpu_data;
    float* gpu_data;
    cudaMallocHost((void**)&cpu_data, max_bytes);

    for (int proc = 0; proc < num_procs; proc++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == proc)
        {
            timectr = 0;
            for (int gpu = 0; gpu < num_gpus; gpu++)
            {
                cudaSetDevice(gpu);
                cudaMalloc((void**)&gpu_data, max_bytes);
                for (int i = 0; i < max_i; i++)
                {
                    bytes = pow(2, i) * sizeof(float);
                    times[timectr++] = timeMemcpy(bytes, cpu_data, gpu_data, cudaMemcpyHostToDevice);
                }
                for (int i = 0; i < max_i; i++)
                {
                    bytes = pow(2, i) * sizeof(float);
                    times[timectr++] = timeMemcpy(bytes, gpu_data, cpu_data, cudaMemcpyDeviceToHost);
                }
                cudaFree(gpu_data);
            }
        }
        else std::fill(times.begin(), times.end(), 0);

        MPI_Reduce(times.data(), max_times.data(), times.size(), MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            timectr = 0;
            for (int gpu = 0; gpu < num_gpus; gpu++)
            {
                printf("CPU %d to GPU %d:\t", proc, gpu);
                for (int i = 0; i < max_i; i++)
                    printf("%2.5f\t", max_times[timectr++]);
                printf("\n");
                printf("GPU %d to CPU %d:\t", gpu, proc);
                for (int i = 0; i < max_i; i++)
                    printf("%2.5f\t", max_times[timectr++]);
                printf("\n");
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    cudaFreeHost(cpu_data);

    MPI_Finalize();
}
