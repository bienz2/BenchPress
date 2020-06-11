#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>

void timeThreeStep(int size, float* cpu_data, float* gpu_data, cudaStream_t& stream, int ppg, int gpu_rank)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    float t0, tfinal;
    int n_tests = 1000;
    for (int np = 0; np < ppg; np++)
    {
        if (size <= np) 
        {
            if (rank == 0) printf("-1\t");
            continue;
        }

        int bytes = size * sizeof(float);

        int procsize = size / (np+1);
        if (procsize > 1000000) n_tests = 100;
        if (procsize > 30000000) n_tests = 10;
        int extra = size % (np + 1);
        int intersize = procsize;
        if (extra > gpu_rank) intersize += 1;

        int ping_test = 1234;
        int pong_test = 4321;
        int intrastep1 = 5678;
        int intrastep2 = 8765;


        cudaDeviceSynchronize();
        cudaStreamSynchronize(stream);
        if (gpu_rank <= np)
        {
            t0 = MPI_Wtime();
            for (int j = 0; j < n_tests; j++)
            {
                if (rank % 2 == 0)
                {
                    if (gpu_rank == 0)
                    {
                        cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
                        cudaStreamSynchronize(stream);
                        int ctr = intersize; 
                        for (int i = 1; i <= np; i++)
                        {
                            int s = procsize;
                            if (extra > i) s += 1;
                            MPI_Send(&(cpu_data[ctr]), s, MPI_FLOAT, rank + (2*i), intrastep1, MPI_COMM_WORLD);
                            ctr += s;
                        }
                    }
                    else
                        MPI_Recv(cpu_data, intersize, MPI_FLOAT, rank - (2*gpu_rank), intrastep1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    MPI_Send(cpu_data, intersize, MPI_FLOAT, rank+1, ping_test, MPI_COMM_WORLD);
                    MPI_Recv(cpu_data, intersize, MPI_FLOAT, rank+1, pong_test, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    if (gpu_rank == 0)
                    {
                        int ctr = intersize;
                        for (int i = 1; i <= np; i++)
                        {
                            int s = procsize;
                            if (extra > i) s += 1;
                            MPI_Recv(&(cpu_data[ctr]), s, MPI_FLOAT, rank + (2*i), intrastep2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            ctr += s;
                        }
                        cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
                        cudaStreamSynchronize(stream);
                    }
                    else
                        MPI_Send(cpu_data, intersize, MPI_FLOAT, rank - (2*gpu_rank), intrastep2, MPI_COMM_WORLD);
                }
                else
                {
                    MPI_Recv(cpu_data, intersize, MPI_FLOAT, rank-1, ping_test, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    if (gpu_rank == 0)
                    {
                        int ctr = intersize;
                        for (int i = 1; i <= np; i++)
                        {
                            int s = procsize;
                            if (extra > i) s += 1;
                            MPI_Recv(&(cpu_data[ctr]), s, MPI_FLOAT, rank + (2*i), intrastep1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            ctr += s;
                        }
                        cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
                        cudaStreamSynchronize(stream);
                    }
                    else
                        MPI_Send(cpu_data, intersize, MPI_FLOAT, rank - (2*gpu_rank), intrastep1, MPI_COMM_WORLD);

                    if (gpu_rank == 0)
                    {
                        cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
                        cudaStreamSynchronize(stream);
                        int ctr = intersize; 
                        for (int i = 1; i <= np; i++)
                        {
                            int s = procsize;
                            if (extra > i) s += 1;
                            MPI_Send(&(cpu_data[ctr]), s, MPI_FLOAT, rank + (2*i), intrastep2, MPI_COMM_WORLD);
                            ctr += s;
                        }
                    }
                    else
                        MPI_Recv(cpu_data, intersize, MPI_FLOAT, rank - (2*gpu_rank), intrastep2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    MPI_Send(cpu_data, intersize, MPI_FLOAT, rank-1, pong_test, MPI_COMM_WORLD);
                }
            }
            tfinal = ((MPI_Wtime() - t0) / n_tests) * 1000;
        }

        cudaDeviceSynchronize();
        cudaStreamSynchronize(stream);
        MPI_Barrier(MPI_COMM_WORLD);

        if (gpu_rank <= np)
        {
            t0 = MPI_Wtime();
            for (int j = 0; j < n_tests; j++)
            {
                if (rank % 2 == 0)
                {
                    if (gpu_rank == 0)
                    {
                        cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
                        cudaStreamSynchronize(stream);
                        int ctr = intersize; 
                        for (int i = 1; i <= np; i++)
                        {
                            int s = procsize;
                            if (extra > i) s += 1;
                            MPI_Send(&(cpu_data[ctr]), s, MPI_FLOAT, rank + (2*i), intrastep1, MPI_COMM_WORLD);
                            ctr += s;
                        }
                    }
                    else
                        MPI_Recv(cpu_data, intersize, MPI_FLOAT, rank - (2*gpu_rank), intrastep1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    MPI_Send(cpu_data, intersize, MPI_FLOAT, rank+1, ping_test, MPI_COMM_WORLD);
                    MPI_Recv(cpu_data, intersize, MPI_FLOAT, rank+1, pong_test, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


                    if (gpu_rank == 0)
                    {
                        int ctr = intersize; 
                        for (int i = 1; i <= np; i++)
                        {
                            int s = procsize;
                            if (extra > i) s += 1;
                            MPI_Recv(&(cpu_data[ctr]), s, MPI_FLOAT, rank + (2*i), intrastep1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            ctr += s;
                        }
                        cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
                        cudaStreamSynchronize(stream);
                    }
                    else
                        MPI_Send(cpu_data, intersize, MPI_FLOAT, rank - (2*gpu_rank), intrastep1, MPI_COMM_WORLD);
                }
                else
                {
                    MPI_Recv(cpu_data, intersize, MPI_FLOAT, rank-1, ping_test, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    if (gpu_rank == 0)
                    {
                        int ctr = intersize;
                        for (int i = 1; i <= np; i++)
                        {
                            int s = procsize;
                            if (extra > i) s += 1;
                            MPI_Recv(&(cpu_data[ctr]), s, MPI_FLOAT, rank + (2*i), intrastep1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            ctr += s;
                        }
                        cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
                        cudaStreamSynchronize(stream);
                    }
                    else
                        MPI_Send(cpu_data, intersize, MPI_FLOAT, rank - (2*gpu_rank), intrastep1, MPI_COMM_WORLD);

                   if (gpu_rank == 0)
                    {
                        cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
                        cudaStreamSynchronize(stream);
                        int ctr = intersize; 
                        for (int i = 1; i <= np; i++)
                        {
                            int s = procsize;
                            if (extra > i) s += 1;
                            MPI_Send(&(cpu_data[ctr]), s, MPI_FLOAT, rank + (2*i), intrastep2, MPI_COMM_WORLD);
                            ctr += s;
                        }
                    }
                    else
                        MPI_Recv(cpu_data, intersize, MPI_FLOAT, rank - (2*gpu_rank), intrastep2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    MPI_Send(cpu_data, intersize, MPI_FLOAT, rank-1, pong_test, MPI_COMM_WORLD);
                }
            }
            tfinal = ((MPI_Wtime() - t0) / n_tests) * 1000;
        }
        else tfinal = 0;

        MPI_Reduce(&tfinal, &t0, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("%e\t", t0);
    }
    if (rank == 0) printf("\n");
}


void timeThreeStepAsync(int size, float* cpu_data, float* gpu_data, cudaStream_t& stream, int ppg, int gpu_rank)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    float t0, tfinal;
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
            t0 = MPI_Wtime();
            for (int j = 0; j < n_tests; j++)
            {
                if (rank % 2 == 0)
                {
                    cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    MPI_Send(cpu_data, procsize, MPI_FLOAT, rank+1, 1234, MPI_COMM_WORLD);
                    MPI_Recv(cpu_data, procsize, MPI_FLOAT, rank+1, 4321, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
                    cudaStreamSynchronize(stream);
                }
                else
                {
                    MPI_Recv(cpu_data, procsize, MPI_FLOAT, rank-1, 1234, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
                    cudaStreamSynchronize(stream);
                    cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    MPI_Send(cpu_data, procsize, MPI_FLOAT, rank-1, 4321, MPI_COMM_WORLD);
                } 
            }
            tfinal = ((MPI_Wtime() - t0) / n_tests) * 1000;
        }

        cudaDeviceSynchronize();
        cudaStreamSynchronize(stream);
        MPI_Barrier(MPI_COMM_WORLD);

        if (gpu_rank <= np)
        {
            t0 = MPI_Wtime();
            for (int j = 0; j < n_tests; j++) 
            {
                if (rank % 2 == 0)
                {
                    cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    MPI_Send(cpu_data, procsize, MPI_FLOAT, rank+1, 1234, MPI_COMM_WORLD);
                    MPI_Recv(cpu_data, procsize, MPI_FLOAT, rank+1, 4321, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
                    cudaStreamSynchronize(stream);
                }
                else
                {
                    MPI_Recv(cpu_data, procsize, MPI_FLOAT, rank-1, 1234, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
                    cudaStreamSynchronize(stream);
                    cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    MPI_Send(cpu_data, procsize, MPI_FLOAT, rank-1, 4321, MPI_COMM_WORLD);
                } 
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


    int node_rank = rank / 2;
    int ppn = 36;
    int pps = ppn / 2;
    int ppg = ppn / num_gpus; 
    int socket_rank = node_rank % pps;
    int gpu = node_rank / ppg;
    int gpu_rank = node_rank % ppg;
    cudaSetDevice(gpu);
    cudaMalloc((void**)&gpu_data, max_bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
//cudaStream_t stream = 0;

    if (rank == 0) printf("3Step:\n");
    for (int i = 0; i < max_i; i++)
    {
        size = pow(2, i);

        if (rank == 0) printf("%d:\t", size);
        timeThreeStep(size, cpu_data, gpu_data, stream, ppg, gpu_rank);
    }
    if (rank == 0) printf("3Step Async:\n");
    for (int i = 0; i < max_i; i++)
    {
        size = pow(2, i);

        if (rank == 0) printf("%d:\t", size);
        timeThreeStepAsync(size, cpu_data, gpu_data, stream, ppg, gpu_rank);
    }
    cudaFree(gpu_data);
    cudaStreamDestroy(stream);
    cudaFreeHost(cpu_data);

    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        printf("ERROR!\n");
        exit( -1 );
    }


    MPI_Finalize();
}
