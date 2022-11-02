#include "ping_pong_timer.h"

double time_ping_pong(bool active, int rank0, int rank1, float* data, 
        int size, int n_tests)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t0, tfinal;
    int ping_tag = 1234;
    int pong_tag = 4321;
    MPI_Status status;

    // Warm Up
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (active)
    {
        t0 = MPI_Wtime();
        for (int i = 0; i < n_tests; i++)
        {
            if (rank == rank0)
            {
                MPI_Send(data, size, MPI_FLOAT, rank1, ping_tag, MPI_COMM_WORLD);
                MPI_Recv(data, size, MPI_FLOAT, rank1, pong_tag, MPI_COMM_WORLD, &status);
            }
            else if (rank == rank1)
            {
                MPI_Recv(data, size, MPI_FLOAT, rank0, ping_tag, MPI_COMM_WORLD, &status);
                MPI_Send(data, size, MPI_FLOAT, rank0, pong_tag, MPI_COMM_WORLD);
            }
        }
        tfinal = (MPI_Wtime() - t0) / (2 * n_tests);
    }
    else tfinal = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    if (active)
    {
        t0 = MPI_Wtime();
        for (int i = 0; i < n_tests; i++)
        {
            if (rank == rank0)
            {
                MPI_Send(data, size, MPI_FLOAT, rank1, ping_tag, MPI_COMM_WORLD);
                MPI_Recv(data, size, MPI_FLOAT, rank1, pong_tag, MPI_COMM_WORLD, &status);
            }
            else if (rank == rank1)
            {
                MPI_Recv(data, size, MPI_FLOAT, rank0, ping_tag, MPI_COMM_WORLD, &status);
                MPI_Send(data, size, MPI_FLOAT, rank0, pong_tag, MPI_COMM_WORLD);
            }
        }
        tfinal = (MPI_Wtime() - t0) / (2 * n_tests);
    }
    else tfinal = 0;

    return tfinal;
}

double time_ping_pong_3step(bool active, int rank0, int rank1, float* cpu_data, 
        float* gpu_data, int size, gpuStream_t stream, int n_tests)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t0, tfinal;
    int ping_tag = 1234;
    int pong_tag = 4321;
    MPI_Status status;
    int bytes = size * sizeof(float);

    // Warm Up
    gpuDeviceSynchronize();
    gpuStreamSynchronize(stream);
    MPI_Barrier(MPI_COMM_WORLD);
    if (active)
    {
        t0 = MPI_Wtime();
        for (int i = 0; i < n_tests; i++)
        {
            if (rank == rank0)
            {
                gpuMemcpyAsync(cpu_data, gpu_data, bytes, gpuMemcpyDeviceToHost, stream);
                gpuStreamSynchronize(stream);
                MPI_Send(cpu_data, size, MPI_FLOAT, rank1, ping_tag, MPI_COMM_WORLD);
                MPI_Recv(cpu_data, size, MPI_FLOAT, rank1, pong_tag, MPI_COMM_WORLD, &status);
                gpuMemcpyAsync(gpu_data, cpu_data, bytes, gpuMemcpyHostToDevice, stream);
                gpuStreamSynchronize(stream);
            }
            else if (rank == rank1)
            {
                MPI_Recv(cpu_data, size, MPI_FLOAT, rank0, ping_tag, MPI_COMM_WORLD, &status);
                gpuMemcpyAsync(gpu_data, cpu_data, bytes, gpuMemcpyHostToDevice, stream);
                gpuStreamSynchronize(stream);
                gpuMemcpyAsync(cpu_data, gpu_data, bytes, gpuMemcpyDeviceToHost, stream);
                gpuStreamSynchronize(stream);
                MPI_Send(cpu_data, size, MPI_FLOAT, rank0, pong_tag, MPI_COMM_WORLD);
            }
        }
        tfinal = (MPI_Wtime() - t0) / (2 * n_tests);
    }
    else tfinal = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    if (active)
    {
        t0 = MPI_Wtime();
        for (int i = 0; i < n_tests; i++)
        {
            if (rank == rank0)
            {
                gpuMemcpyAsync(cpu_data, gpu_data, bytes, gpuMemcpyDeviceToHost, stream);
                gpuStreamSynchronize(stream);
                MPI_Send(cpu_data, size, MPI_FLOAT, rank1, ping_tag, MPI_COMM_WORLD);
                MPI_Recv(cpu_data, size, MPI_FLOAT, rank1, pong_tag, MPI_COMM_WORLD, &status);
                gpuMemcpyAsync(gpu_data, cpu_data, bytes, gpuMemcpyHostToDevice, stream);
                gpuStreamSynchronize(stream);
            }
            else if (rank == rank1)
            {
                MPI_Recv(cpu_data, size, MPI_FLOAT, rank0, ping_tag, MPI_COMM_WORLD, &status);
                gpuMemcpyAsync(gpu_data, cpu_data, bytes, gpuMemcpyHostToDevice, stream);
                gpuStreamSynchronize(stream);
                gpuMemcpyAsync(cpu_data, gpu_data, bytes, gpuMemcpyDeviceToHost, stream);
                gpuStreamSynchronize(stream);
                MPI_Send(cpu_data, size, MPI_FLOAT, rank0, pong_tag, MPI_COMM_WORLD);
            }
        }
        tfinal = (MPI_Wtime() - t0) / (2 * n_tests);
    }
    else tfinal = 0;

    return tfinal;
}

double time_ping_pong_mult(bool master, int n_msgs, int* procs,
        float* data, int size, int n_tests)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t0, tfinal;
    int ping_tag = 1234;
    int pong_tag = 4321;
    int proc;
    MPI_Status status;

    // Warm Up
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        if (master)
        {
            for (int j = 0; j < n_msgs; j++)
            {
                proc = procs[j];
                MPI_Send(&(data[j*size]), size, MPI_FLOAT, proc, ping_tag, MPI_COMM_WORLD);
            }
            for (int j = 0; j < n_msgs; j++)
            { 
                proc = procs[j];
                MPI_Recv(&(data[j*size]), size, MPI_FLOAT, proc, pong_tag, MPI_COMM_WORLD, 
                        &status);
            }
        }
        else
        {
            for (int j = 0; j < n_msgs; j++)
            {
                proc = procs[j];
                MPI_Recv(&(data[j*size]), size, MPI_FLOAT, proc, ping_tag, MPI_COMM_WORLD,
                        &status);
            }
            for (int j = 0; j < n_msgs; j++)
            {
                proc = procs[j];
                MPI_Send(&(data[j*size]), size, MPI_FLOAT, proc, pong_tag, MPI_COMM_WORLD);
            }
        }
    }
    tfinal = (MPI_Wtime() - t0) / (2 * n_tests);

    // Time Ping-Pong
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        if (master)
        {
            for (int j = 0; j < n_msgs; j++)
            {
                proc = procs[j];
                MPI_Send(&(data[j*size]), size, MPI_FLOAT, proc, ping_tag, MPI_COMM_WORLD);
            }
            for (int j = 0; j < n_msgs; j++)
            {
                proc = procs[j];
                MPI_Recv(&(data[j*size]), size, MPI_FLOAT, proc, pong_tag, MPI_COMM_WORLD,
                        &status);
            }
        }
        else
        {
            for (int j = 0; j < n_msgs; j++)
            {
                proc = procs[j];
                MPI_Recv(&(data[j*size]), size, MPI_FLOAT, proc, ping_tag, MPI_COMM_WORLD,
                    &status);
            }
            for (int j = 0; j < n_msgs; j++)
            {
                proc = procs[j];
                MPI_Send(&(data[j*size]), size, MPI_FLOAT, proc, pong_tag, MPI_COMM_WORLD);
            }
        }
    }
    tfinal = (MPI_Wtime() - t0) / (2 * n_tests);

    return tfinal;
}

        

