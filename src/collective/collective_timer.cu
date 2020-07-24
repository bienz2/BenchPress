#include "collective_timer.h"

double time_allreduce(int size, float* gpu_data, MPI_Comm& group_comm,
        int n_tests)
{
    double t0, tfinal;

    // Warm Up
    cudaDeviceSynchronize();
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        MPI_Allreduce(MPI_IN_PLACE, gpu_data, size, MPI_FLOAT, MPI_MAX, group_comm);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;


    cudaDeviceSynchronize();
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        MPI_Allreduce(MPI_IN_PLACE, gpu_data, size, MPI_FLOAT, MPI_MAX, group_comm);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;

    return tfinal;
}

double time_allreduce_3step(int size, float* cpu_data, float* gpu_data,
        cudaStream_t& stream, MPI_Comm& group_comm, int n_tests)
{
    double t0, tfinal;
    int bytes = size * sizeof(float);

    // Warm Up
    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        MPI_Allreduce(MPI_IN_PLACE, cpu_data, size, MPI_FLOAT, MPI_MAX, group_comm);
        cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;


    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        MPI_Allreduce(MPI_IN_PLACE, cpu_data, size, MPI_FLOAT, MPI_MAX, group_comm);
        cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;

    return tfinal;
}

double time_allreduce_3step_msg(int size, float* cpu_data, float* gpu_data,
       int ppg, int node_rank, cudaStream_t& stream, MPI_Comm& group_comm, 
       int n_tests)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double t0, tfinal;
    int bytes = size * sizeof(float);
    int msg_size = size / ppg;
    bool master = node_rank % ppg == 0;
    int ping_tag = 1234;
    int pong_tag = 4321;
    MPI_Status status;

    // Warm Up 
    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        if (master)
        {
            cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            for (int i = 1; i < ppg; i++)
            {
                MPI_Send(&(cpu_data[msg_size*i]), msg_size, MPI_FLOAT, rank+i, ping_tag, 
                        MPI_COMM_WORLD);
            }
        }
        else
        {
            MPI_Recv(cpu_data, msg_size, MPI_FLOAT, rank - node_rank, ping_tag, MPI_COMM_WORLD, &status);
        }

        MPI_Allreduce(MPI_IN_PLACE, cpu_data, msg_size, MPI_FLOAT, MPI_MAX, group_comm);
      
        if (master)
        {
            for (int i = 1; i < ppg; i++)
            {
                MPI_Recv(&(cpu_data[msg_size*i]), msg_size, MPI_FLOAT, rank+i, pong_tag,
                       MPI_COMM_WORLD, &status);
            }
            cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
        }
        else
        {
           MPI_Send(cpu_data, msg_size, MPI_FLOAT, rank - node_rank, pong_tag,
                  MPI_COMM_WORLD); 
        }
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;
    

    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        if (master)
        {
            cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            for (int i = 1; i < ppg; i++)
            {
                MPI_Send(&(cpu_data[msg_size*i]), msg_size, MPI_FLOAT, rank+i, ping_tag, 
                        MPI_COMM_WORLD);
            }
        }
        else
        {
            MPI_Recv(cpu_data, msg_size, MPI_FLOAT, rank - node_rank, ping_tag, MPI_COMM_WORLD, &status);
        }

        MPI_Allreduce(MPI_IN_PLACE, cpu_data, msg_size, MPI_FLOAT, MPI_MAX, group_comm);
      
        if (master)
        {
            for (int i = 1; i < ppg; i++)
            {
                MPI_Recv(&(cpu_data[msg_size*i]), msg_size, MPI_FLOAT, rank+i, pong_tag,
                       MPI_COMM_WORLD, &status);
            }
            cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
        }
        else
        {
           MPI_Send(cpu_data, msg_size, MPI_FLOAT, rank - node_rank, pong_tag,
                  MPI_COMM_WORLD); 
        }
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;

    return tfinal;
}

