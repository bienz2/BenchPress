#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>

float timeMemcpy(int bytes, float* orig_x, float* dest_x,
        cudaMemcpyKind copy_kind, cudaStream_t stream, 
        int n_tests = 1000)
{
    float time;
    cudaEvent_t startEvent, stopEvent;

    // Warm Up
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    cudaEventRecord(startEvent, stream);
    for (int i = 0; i < n_tests; i++)
    {
        cudaMemcpyAsync(dest_x, orig_x, bytes, copy_kind, stream);
        cudaStreamSynchronize(stream);
    }
    cudaEventRecord(stopEvent, stream);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    // Time Memcpy
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    cudaEventRecord(startEvent, stream);
    for (int i = 0; i < n_tests; i++)
    {
        cudaMemcpyAsync(dest_x, orig_x, bytes, copy_kind, stream);
        cudaStreamSynchronize(stream);
    }
    cudaEventRecord(stopEvent, stream);
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

    int max_i = 24;
    int max_bytes = pow(2,max_i-1) * sizeof(float);
    int n_times = 2 * max_i * num_gpus;
    int timectr, bytes;
    std::vector<float> times(n_times);
    std::vector<float> max_times(n_times);
    float* cpu_data;
    float* gpu_data;
    int n_tests;
    cudaMallocHost((void**)&cpu_data, max_bytes);


//    for (int proc = 0; proc < num_procs; proc++)
    for (int proc = 0; proc < 4; proc++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == proc)
        {

            timectr = 0;
            for (int gpu = 0; gpu < num_gpus; gpu++)
            {
                cudaSetDevice(gpu);
                cudaMalloc((void**)&gpu_data, max_bytes);
                cudaStream_t proc_stream;
                cudaStreamCreate(&proc_stream);
                n_tests = 1000;
                for (int i = 0; i < max_i; i++)
                {
                    if (i > 20) n_tests = 100;
                    if (i > 25) n_tests = 10;
                    bytes = pow(2, i) * sizeof(float);
                    times[timectr++] = timeMemcpy(bytes, cpu_data, gpu_data, cudaMemcpyHostToDevice, proc_stream, n_tests);
                }
                n_tests = 1000;
                for (int i = 0; i < max_i; i++)
                {
                    if (i > 20) n_tests = 100;
                    if (i > 25) n_tests = 10;
                    bytes = pow(2, i) * sizeof(float);
                    times[timectr++] = timeMemcpy(bytes, gpu_data, cpu_data, cudaMemcpyDeviceToHost, proc_stream, n_tests);
                }
                cudaFree(gpu_data);
                cudaStreamDestroy(proc_stream);
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
