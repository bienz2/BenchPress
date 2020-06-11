#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>

void time_memcpy(int size, float* orig_x, float* dest_x, cudaMemcpyKind copy_kind, cudaStream_t& stream,
        int gpu, int ppg, int gpu_rank)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    float t0, tfinal;

    cudaStreamSynchronize(stream);

    int n_tests = 1000;
    for (int np = 0; np < ppg; np++)
    {
        if (size <= np) 
        {
            if (rank == 0) printf("-1\t");
            continue;
        }

        int procsize = size / (np+1);
        if (procsize > 1000000) n_tests = 100;
        if (procsize > 30000000) n_tests = 10;
        int extra = size % (np + 1);
        if (gpu_rank < extra) procsize += 1;
        int bytes = procsize * sizeof(float);

        cudaDeviceSynchronize();
        cudaStreamSynchronize(stream);
 
         if (gpu_rank <= np)
        {
            for (int j = 0; j < n_tests; j++)
            {
                cudaMemcpyAsync(dest_x, orig_x, bytes, copy_kind, stream);
                cudaStreamSynchronize(stream);
            }
        }

        cudaDeviceSynchronize();
        cudaStreamSynchronize(stream);
        MPI_Barrier(MPI_COMM_WORLD);

        if (gpu_rank <= np)
        {
            t0 = MPI_Wtime();
            for (int j = 0; j < n_tests; j++) 
            {
                cudaMemcpyAsync(dest_x, orig_x, bytes, copy_kind, stream);
                cudaStreamSynchronize(stream);
            }
            tfinal = ((MPI_Wtime() - t0) / n_tests) * 1000;
        }
        else tfinal = 0;

        MPI_Reduce(&tfinal, &t0, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("%e\t", t0);
    }
    if (rank == 0) printf("\n");
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
    int size, procsize, bytes;
    std::vector<float> times(n_times);
    std::vector<float> max_times(n_times);
    float* cpu_data;
    float* gpu_data;
    int n_tests;
    cudaMallocHost((void**)&cpu_data, max_bytes);


    int node_rank = rank;
    int ppn = 36;
    int pps = ppn / 2;
    int ppg = ppn / num_gpus; 
    int socket_rank = node_rank % pps;
    int gpu = node_rank / ppg;
    int gpu_rank = node_rank % ppg;
    cudaSetDevice(gpu);
    cudaMalloc((void**)&gpu_data, max_bytes);

//    cudaStream_t stream;
 //   cudaStreamCreate(&stream);
cudaStream_t stream = 0;

    if (rank == 0) printf("H2D:\n");
    for (int i = 0; i < max_i; i++)
    {
        size = pow(2, i);

        if (rank == 0) printf("%d:\t", size);
        time_memcpy(size, cpu_data, gpu_data, cudaMemcpyHostToDevice, stream, gpu, ppg, gpu_rank);
    }
    if (rank == 0) printf("D2H:\n");
    for (int i = 0; i < max_i; i++)
    {
        size = pow(2, i);

        if (rank == 0) printf("%d:\t", size);
        time_memcpy(size, gpu_data, cpu_data, cudaMemcpyDeviceToHost, stream, gpu, ppg, gpu_rank);
    }

//    cudaStreamDestroy(stream);
    cudaFree(gpu_data);
    cudaFreeHost(cpu_data);


    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        printf("ERROR!\n");
        exit( -1 );
    }


    MPI_Finalize();
}
