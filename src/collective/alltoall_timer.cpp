#include "alltoall_timer.hpp"

double time_alltoall(int size, float* gpu_data, MPI_Comm& group_comm,
        int n_tests)
{
    int  num_procs;
    MPI_Comm_size(group_comm, &num_procs);

    double t0, tfinal;

    // Warm Up
    gpuDeviceSynchronize();
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        MPI_Alltoall(MPI_IN_PLACE, size, MPI_FLOAT, gpu_data, size, MPI_FLOAT,
                group_comm);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;


    gpuDeviceSynchronize();
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        MPI_Alltoall(MPI_IN_PLACE, size, MPI_FLOAT, gpu_data, size, MPI_FLOAT,
                group_comm);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;

    return tfinal;
}

double time_alltoall_3step(int size, float* cpu_data, float* gpu_data,
        gpuStream_t& stream, MPI_Comm& group_comm, int n_tests)
{
    int num_procs;
    MPI_Comm_size(group_comm, &num_procs);

    double t0, tfinal;
    int total_size = size * num_procs;
    int bytes = total_size * sizeof(float);

    // Warm Up
    gpuDeviceSynchronize();
    gpuStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        gpuMemcpyAsync(cpu_data, gpu_data, bytes, gpuMemcpyDeviceToHost, stream);
        gpuStreamSynchronize(stream);
        MPI_Alltoall(MPI_IN_PLACE, size, MPI_FLOAT, cpu_data, size, MPI_FLOAT,
                group_comm);
        gpuMemcpyAsync(gpu_data, cpu_data, bytes, gpuMemcpyHostToDevice, stream);
        gpuStreamSynchronize(stream);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;


    gpuDeviceSynchronize();
    gpuStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        gpuMemcpyAsync(cpu_data, gpu_data, bytes, gpuMemcpyDeviceToHost, stream);
        gpuStreamSynchronize(stream);
        MPI_Alltoall(MPI_IN_PLACE, size, MPI_FLOAT, cpu_data, size, MPI_FLOAT,
                group_comm);
        gpuMemcpyAsync(gpu_data, cpu_data, bytes, gpuMemcpyHostToDevice, stream);
        gpuStreamSynchronize(stream);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;

    return tfinal;
}

double time_alltoall_3step_msg(int size, float* cpu_data, float* gpu_data,
       int ppg, int node_rank, gpuStream_t& stream, MPI_Comm& group_comm, 
       int n_tests)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(group_comm, &num_procs);

    double t0, tfinal;
    int total_size = size * num_procs;
    int bytes = total_size * sizeof(float);
    int msg_size = total_size / ppg;
    int coll_size = size / ppg;
    int gpu_rank = node_rank % ppg;
    bool master = gpu_rank == 0;
    int ping_tag = 1234;
    int pong_tag = 4321;
    MPI_Status status;

    // Warm Up 
    gpuDeviceSynchronize();
    gpuStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        if (master)
        {
            gpuMemcpyAsync(cpu_data, gpu_data, bytes, gpuMemcpyDeviceToHost, stream);
            gpuStreamSynchronize(stream);
            for (int i = 1; i < ppg; i++)
            {
                MPI_Send(&(cpu_data[msg_size*i]), msg_size, MPI_FLOAT, rank+i, 
                        ping_tag, MPI_COMM_WORLD);
            }
        }
        else
        {
            MPI_Recv(cpu_data, msg_size, MPI_FLOAT, rank - gpu_rank, 
                    ping_tag, MPI_COMM_WORLD, &status);
        }

        MPI_Alltoall(MPI_IN_PLACE, coll_size, MPI_FLOAT, cpu_data, coll_size, MPI_FLOAT,
                group_comm);
      
        if (master)
        {
            for (int i = 1; i < ppg; i++)
            {
                MPI_Recv(&(cpu_data[msg_size*i]), msg_size, MPI_FLOAT, rank+i, 
                       pong_tag, MPI_COMM_WORLD, &status);
            }
            gpuMemcpyAsync(gpu_data, cpu_data, bytes, gpuMemcpyHostToDevice, stream);
            gpuStreamSynchronize(stream);
        }
        else
        {
           MPI_Send(cpu_data, msg_size, MPI_FLOAT, rank - gpu_rank, 
                  pong_tag, MPI_COMM_WORLD); 
        }
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;
    

    gpuDeviceSynchronize();
    gpuStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        if (master)
        {
            gpuMemcpyAsync(cpu_data, gpu_data, bytes, gpuMemcpyDeviceToHost, stream);
            gpuStreamSynchronize(stream);
            for (int i = 1; i < ppg; i++)
            {
                MPI_Send(&(cpu_data[msg_size*i]), msg_size, MPI_FLOAT, rank+i, 
                        ping_tag, MPI_COMM_WORLD);
            }
        }
        else
        {
            MPI_Recv(cpu_data, msg_size, MPI_FLOAT, rank - gpu_rank, 
                    ping_tag, MPI_COMM_WORLD, &status);
        }

        MPI_Alltoall(MPI_IN_PLACE, coll_size, MPI_FLOAT, cpu_data, coll_size, MPI_FLOAT,
                group_comm);
      
        if (master)
        {
            for (int i = 1; i < ppg; i++)
            {
                MPI_Recv(&(cpu_data[msg_size*i]), msg_size, MPI_FLOAT, rank+i, 
                       pong_tag, MPI_COMM_WORLD, &status);
            }
            gpuMemcpyAsync(gpu_data, cpu_data, bytes, gpuMemcpyHostToDevice, stream);
            gpuStreamSynchronize(stream);
        }
        else
        {
           MPI_Send(cpu_data, msg_size, MPI_FLOAT, rank - gpu_rank, pong_tag,
                  MPI_COMM_WORLD); 
        }
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;

    return tfinal;
}

