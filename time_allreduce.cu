#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>

void timeGPU(int size, float* gpu_data, int n_tests = 1000)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    float t0, tfinal;
    tfinal = 0;
    cudaDeviceSynchronize();
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
        MPI_Allreduce(MPI_IN_PLACE, gpu_data, size, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    tfinal = ((MPI_Wtime() - t0) / n_tests) * 1000;


    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
        MPI_Allreduce(MPI_IN_PLACE, gpu_data, size, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    tfinal = ((MPI_Wtime() - t0) / n_tests) * 1000;

    MPI_Reduce(&tfinal, &t0, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%e\t", t0);
}

void timeCPU(int size, float* cpu_data, float* gpu_data, cudaStream_t& stream, int gpu_rank, MPI_Comm& group_comm, int n_tests = 1000)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    float t0, tfinal;
    int bytes = size * sizeof(float);
 
    if (gpu_rank == 0)
    {
        cudaDeviceSynchronize();
        t0 = MPI_Wtime();
        for (int i = 0; i < n_tests; i++)
        {
            cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            MPI_Allreduce(MPI_IN_PLACE, cpu_data, size, MPI_FLOAT, MPI_MAX, group_comm);
            cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
        }
        tfinal = ((MPI_Wtime() - t0) / n_tests) * 1000;
        cudaDeviceSynchronize();
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (gpu_rank == 0)
    {
        t0 = MPI_Wtime();
        for (int i = 0; i < n_tests; i++)
        {
            cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            MPI_Allreduce(MPI_IN_PLACE, cpu_data, size, MPI_FLOAT, MPI_MAX, group_comm);
            cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
        }
        tfinal = ((MPI_Wtime() - t0) / n_tests) * 1000;
    }
    else tfinal = 0;

    MPI_Reduce(&tfinal, &t0, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%e\t", t0);
}

void timeThreeStep(int size, float* cpu_data, float* gpu_data, cudaStream_t& stream, std::vector<int>& proc_list, int num_nodes, int gpu_rank, MPI_Comm& group_comm, int n_tests = 1000)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    float t0, tfinal;
    for (int np = 0; np < proc_list.size(); np++)
    {
        int ppg = proc_list[np];
        if (size < ppg) 
        {
            if (rank == 0) printf("-1\t");
            continue;
        }

        int bytes = size * sizeof(float);
        int proc_size = size / ppg;
        
        int inittag = 1234;
        int desttag = 4321;

        cudaDeviceSynchronize();
        cudaStreamSynchronize(stream);
        if (gpu_rank < ppg)
        {
            t0 = MPI_Wtime();
            for (int j = 0; j < n_tests; j++)
            {
                if (gpu_rank == 0)
                {
                    cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    for (int i = 1; i < ppg; i++)
                        MPI_Send(&(cpu_data[i*proc_size]), proc_size, MPI_FLOAT, rank + (num_nodes*i), inittag, MPI_COMM_WORLD);
                }
                else
                    MPI_Recv(cpu_data, proc_size, MPI_FLOAT, rank - (num_nodes*gpu_rank), inittag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                MPI_Allreduce(MPI_IN_PLACE, cpu_data, proc_size, MPI_FLOAT, MPI_MAX, group_comm);

                if (gpu_rank == 0)
                {
                    for (int i = 1; i < ppg; i++)
                        MPI_Recv(&(cpu_data[i*proc_size]), proc_size, MPI_FLOAT, rank + (num_nodes*i), desttag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
                    cudaStreamSynchronize(stream);
                }
                else
                    MPI_Send(cpu_data, proc_size, MPI_FLOAT, rank - (num_nodes*gpu_rank), desttag, MPI_COMM_WORLD);
            }
            tfinal = ((MPI_Wtime() - t0) / n_tests) * 1000;
        }

        cudaDeviceSynchronize();
        cudaStreamSynchronize(stream);
        MPI_Barrier(MPI_COMM_WORLD);

        if (gpu_rank < ppg)
        {
            t0 = MPI_Wtime();
            for (int j = 0; j < n_tests; j++)
            {
                if (gpu_rank == 0)
                {
                    cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    for (int i = 1; i < ppg; i++)
                        MPI_Send(&(cpu_data[i*proc_size]), proc_size, MPI_FLOAT, rank + (num_nodes*i), inittag, MPI_COMM_WORLD);
                }
                else
                    MPI_Recv(cpu_data, proc_size, MPI_FLOAT, rank - (num_nodes*gpu_rank), inittag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                MPI_Allreduce(MPI_IN_PLACE, cpu_data, proc_size, MPI_FLOAT, MPI_MAX, group_comm);

                if (gpu_rank == 0)
                {
                    for (int i = 1; i < ppg; i++)
                        MPI_Recv(&(cpu_data[i*proc_size]), proc_size, MPI_FLOAT, rank + (num_nodes*i), desttag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
                    cudaStreamSynchronize(stream);
                }
                else
                    MPI_Send(cpu_data, proc_size, MPI_FLOAT, rank - (num_nodes*gpu_rank), desttag, MPI_COMM_WORLD);
            }
            tfinal = ((MPI_Wtime() - t0) / n_tests) * 1000;
        }
        else tfinal = 0;

        MPI_Reduce(&tfinal, &t0, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("%e\t", t0);

    }
    if (rank == 0) printf("\n");
}


void timeThreeStepAsync(int size, float* cpu_data, float* gpu_data, cudaStream_t& stream, std::vector<int>& proc_list, int gpu_rank, MPI_Comm& group_comm, int n_tests = 1000)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
    float t0, tfinal;
    for (int np = 0; np < proc_list.size(); np++)
    {
        int ppg = proc_list[np];

        if (size < ppg) 
        {
            if (rank == 0) printf("-1\t");
            continue;
        }

        int proc_size = size / ppg;
        int bytes = proc_size * sizeof(float);

        cudaDeviceSynchronize();
        cudaStreamSynchronize(stream);
        if (gpu_rank < ppg)
        {
            t0 = MPI_Wtime();
            for (int j = 0; j < n_tests; j++)
            {
                cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                MPI_Allreduce(MPI_IN_PLACE, cpu_data, proc_size, MPI_FLOAT, MPI_MAX, group_comm);
                cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
                cudaStreamSynchronize(stream);
            }
            tfinal = ((MPI_Wtime() - t0) / n_tests) * 1000;
        }

        cudaDeviceSynchronize();
        cudaStreamSynchronize(stream);
        MPI_Barrier(MPI_COMM_WORLD);

        if (gpu_rank < ppg)
        {
            t0 = MPI_Wtime();
            for (int j = 0; j < n_tests; j++)
            {
                cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                MPI_Allreduce(MPI_IN_PLACE, cpu_data, proc_size, MPI_FLOAT, MPI_MAX, group_comm);
                cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
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
    int n_tests = 1000;
    cudaMallocHost((void**)&cpu_data, max_bytes);

    int ppn = 36;
    int num_nodes = num_procs / ppn;
    int node_rank = rank / num_nodes;
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

    std::vector<int> proc_list;
    proc_list.push_back(1);
    proc_list.push_back(2);
    proc_list.push_back(4);
    proc_list.push_back(8);

    MPI_Comm group_comm;
    MPI_Comm_split(MPI_COMM_WORLD, gpu_rank, rank, &group_comm);

    if (rank == 0) printf("GPU Allreduce:\n");
    for (int i = 0; i < max_i; i++)
    {
if (i > 14) n_tests = 100;
if (i > 20) n_tests = 10;
        size = pow(2, i);
        if (rank == 0) printf("%d:\t", size);
        timeGPU(size, gpu_data, n_tests);
        if (rank == 0) printf("\n");
    }
    if (rank == 0) printf("\n");



    if (rank == 0) printf("CPU Allreduce:\n");
    for (int i = 0; i < max_i; i++)
    {
if (i > 14) n_tests = 100;
if (i > 20) n_tests = 10;
        size = pow(2, i);
        if (rank == 0) printf("%d:\t", size);
        timeCPU(size, cpu_data, gpu_data, stream, gpu_rank, group_comm, n_tests);
        if (rank == 0) printf("\n");
    }
    if (rank == 0) printf("\n");



    if (rank == 0) printf("3Step:\n");
    for (int i = 0; i < max_i; i++)
    {
if (i > 14) n_tests = 100;
if (i > 20) n_tests = 10;
        size = pow(2, i);

        if (rank == 0) printf("%d:\t", size);
        timeThreeStep(size, cpu_data, gpu_data, stream, proc_list, num_nodes, gpu_rank, group_comm, n_tests);
    }
    if (rank == 0) printf("\n");



    if (rank == 0) printf("3Step Async:\n");
    for (int i = 0; i < max_i; i++)
    {
if (i > 14) n_tests = 100;
if (i > 20) n_tests = 10;
        size = pow(2, i);

        if (rank == 0) printf("%d:\t", size);
        timeThreeStepAsync(size, cpu_data, gpu_data, stream, proc_list, gpu_rank, group_comm, n_tests);
    }


    cudaFree(gpu_data);
    cudaStreamDestroy(stream);
    cudaFreeHost(cpu_data);
    MPI_Comm_free(&group_comm);

    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        printf("ERROR!\n");
        exit( -1 );
    }


    MPI_Finalize();
}
