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

double time_high_volume_ping_pong(bool active, int rank0, int rank1, float* data, 
        int size, int n_tests, int n_msgs)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t0, tfinal;
    int *ping_tags = new int [n_msgs];
    int *pong_tags = new int [n_msgs];
    int ping_tag = 1234;
    int pong_tag = 4321;
    MPI_Request request[n_msgs];
    MPI_Status  status[n_msgs];

    // Set Tags 
    for (int i = 0; i < n_msgs; i++)
    {
        ping_tags[i] = ping_tag + i;
        pong_tags[i] = pong_tag + i;
    }

    // Warm Up
    MPI_Barrier(MPI_COMM_WORLD);
    if (active)
    {
        t0 = MPI_Wtime();
        for (int i = 0; i < n_tests; i++)
        {
            if (rank == rank0)
            {
                for (int j = 0 ; j < n_msgs; j++)
                {
                    MPI_Isend(data, size, MPI_FLOAT, rank1, ping_tags[j], MPI_COMM_WORLD, &request[j]);
                }
                MPI_Waitall(n_msgs, request, status);

                for (int j = 0 ; j < n_msgs; j++)
                {
                    MPI_Irecv(data, size, MPI_FLOAT, rank1, pong_tags[j], MPI_COMM_WORLD, &request[j]);
                }
                MPI_Waitall(n_msgs, request, status);
            }
            else if (rank == rank1)
            {
                for (int j = 0 ; j < n_msgs; j++)
                {
                    MPI_Irecv(data, size, MPI_FLOAT, rank0, ping_tags[j], MPI_COMM_WORLD, &request[j]);
                }
                MPI_Waitall(n_msgs, request, status);

                for (int j = 0 ; j < n_msgs; j++)
                {
                    MPI_Isend(data, size, MPI_FLOAT, rank0, pong_tags[j], MPI_COMM_WORLD, &request[j]);
                }
                MPI_Waitall(n_msgs, request, status);
            }
        }
        tfinal = (MPI_Wtime() - t0) / (2 * n_tests);
    }
    else tfinal = 0;

     
    // High Volume Ping Pong
    MPI_Barrier(MPI_COMM_WORLD);
    if (active)
    {
        t0 = MPI_Wtime();
        for (int i = 0; i < n_tests; i++)
        {
            if (rank == rank0)
            {
                for (int j = 0 ; j < n_msgs; j++)
                {
                    MPI_Isend(data, size, MPI_FLOAT, rank1, ping_tags[j], MPI_COMM_WORLD, &request[j]);
                }
                MPI_Waitall(n_msgs, request, status);

                for (int j = 0 ; j < n_msgs; j++)
                {
                    MPI_Irecv(data, size, MPI_FLOAT, rank1, pong_tags[j], MPI_COMM_WORLD, &request[j]);
                }
                MPI_Waitall(n_msgs, request, status);
            }
            else if (rank == rank1)
            {
                for (int j = 0 ; j < n_msgs; j++)
                {
                    MPI_Irecv(data, size, MPI_FLOAT, rank0, ping_tags[j], MPI_COMM_WORLD, &request[j]);
                }
                MPI_Waitall(n_msgs, request, status);

                for (int j = 0 ; j < n_msgs; j++)
                {
                    MPI_Isend(data, size, MPI_FLOAT, rank0, pong_tags[j], MPI_COMM_WORLD, &request[j]);
                }
                MPI_Waitall(n_msgs, request, status);
            }
        }
        tfinal = (MPI_Wtime() - t0) / (2 * n_tests);
    }
    else tfinal = 0;

    delete ping_tags;
    delete pong_tags;
    return tfinal;
}

double time_ping_pong_3step(bool active, int rank0, int rank1, float* cpu_data, 
        float* gpu_data, int size, cudaStream_t stream, int n_tests)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t0, tfinal;
    int ping_tag = 1234;
    int pong_tag = 4321;
    MPI_Status status;
    int bytes = size * sizeof(float);

    // Warm Up
    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    MPI_Barrier(MPI_COMM_WORLD);
    if (active)
    {
        t0 = MPI_Wtime();
        for (int i = 0; i < n_tests; i++)
        {
            if (rank == rank0)
            {
                cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                MPI_Send(cpu_data, size, MPI_FLOAT, rank1, ping_tag, MPI_COMM_WORLD);
                MPI_Recv(cpu_data, size, MPI_FLOAT, rank1, pong_tag, MPI_COMM_WORLD, &status);
                cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
                cudaStreamSynchronize(stream);
            }
            else if (rank == rank1)
            {
                MPI_Recv(cpu_data, size, MPI_FLOAT, rank0, ping_tag, MPI_COMM_WORLD, &status);
                cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
                cudaStreamSynchronize(stream);
                cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
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
                cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                MPI_Send(cpu_data, size, MPI_FLOAT, rank1, ping_tag, MPI_COMM_WORLD);
                MPI_Recv(cpu_data, size, MPI_FLOAT, rank1, pong_tag, MPI_COMM_WORLD, &status);
                cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
                cudaStreamSynchronize(stream);
            }
            else if (rank == rank1)
            {
                MPI_Recv(cpu_data, size, MPI_FLOAT, rank0, ping_tag, MPI_COMM_WORLD, &status);
                cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
                cudaStreamSynchronize(stream);
                cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                MPI_Send(cpu_data, size, MPI_FLOAT, rank0, pong_tag, MPI_COMM_WORLD);
            }
        }
        tfinal = (MPI_Wtime() - t0) / (2 * n_tests);
    }
    else tfinal = 0;

    return tfinal;
}

double time_high_volume_ping_pong_3step(bool active, int rank0, int rank1, float* cpu_data, 
        float* gpu_data, int size, cudaStream_t stream, int n_tests, int n_msgs)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t0, tfinal;
    int *ping_tags = new int [n_msgs];
    int *pong_tags = new int [n_msgs];
    int ping_tag = 1234;
    int pong_tag = 4321;
    MPI_Request request[n_msgs];
    MPI_Status  status[n_msgs];
    int bytes = size * sizeof(float);

    // Set Tags 
    for (int i = 0; i < n_msgs; i++)
    {
        ping_tags[i] = ping_tag + i;
        pong_tags[i] = pong_tag + i;
    }

    // Warm Up
    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    MPI_Barrier(MPI_COMM_WORLD);
    if (active)
    {
        t0 = MPI_Wtime();
        for (int i = 0; i < n_tests; i++)
        {
            if (rank == rank0)
            {
                for (int j = 0 ; j < n_msgs; j++)
                {
                    cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    MPI_Isend(cpu_data, size, MPI_FLOAT, rank1, ping_tags[j], MPI_COMM_WORLD, &request[j]);
                }
                MPI_Waitall(n_msgs, request, status);

                for (int j = 0 ; j < n_msgs; j++)
                {
                    MPI_Irecv(cpu_data, size, MPI_FLOAT, rank1, pong_tags[j], MPI_COMM_WORLD, &request[j]);
                    cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
                    cudaStreamSynchronize(stream);
                }
                MPI_Waitall(n_msgs, request, status);
            }
            else if (rank == rank1)
            {
                for (int j = 0 ; j < n_msgs; j++)
                {
                    MPI_Irecv(cpu_data, size, MPI_FLOAT, rank0, ping_tags[j], MPI_COMM_WORLD, &request[j]);
                    cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
                    cudaStreamSynchronize(stream);
                }
                MPI_Waitall(n_msgs, request, status);

                for (int j = 0 ; j < n_msgs; j++)
                {
                    cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    MPI_Isend(cpu_data, size, MPI_FLOAT, rank0, pong_tags[j], MPI_COMM_WORLD, &request[j]);
                }
                MPI_Waitall(n_msgs, request, status);
            }
        }
        tfinal = (MPI_Wtime() - t0) / (2 * n_tests);
    }
    else tfinal = 0;

     
    // High Volume Ping Pong
    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    MPI_Barrier(MPI_COMM_WORLD);
    if (active)
    {
        t0 = MPI_Wtime();
        for (int i = 0; i < n_tests; i++)
        {
            if (rank == rank0)
            {
                for (int j = 0 ; j < n_msgs; j++)
                {
                    cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    MPI_Isend(cpu_data, size, MPI_FLOAT, rank1, ping_tags[j], MPI_COMM_WORLD, &request[j]);
                }
                MPI_Waitall(n_msgs, request, status);

                for (int j = 0 ; j < n_msgs; j++)
                {
                    MPI_Irecv(cpu_data, size, MPI_FLOAT, rank1, pong_tags[j], MPI_COMM_WORLD, &request[j]);
                    cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
                    cudaStreamSynchronize(stream);
                }
                MPI_Waitall(n_msgs, request, status);
            }
            else if (rank == rank1)
            {
                for (int j = 0 ; j < n_msgs; j++)
                {
                    MPI_Irecv(cpu_data, size, MPI_FLOAT, rank0, ping_tags[j], MPI_COMM_WORLD, &request[j]);
                    cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
                    cudaStreamSynchronize(stream);
                }
                MPI_Waitall(n_msgs, request, status);

                for (int j = 0 ; j < n_msgs; j++)
                {
                    cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    MPI_Isend(cpu_data, size, MPI_FLOAT, rank0, pong_tags[j], MPI_COMM_WORLD, &request[j]);
                }
                MPI_Waitall(n_msgs, request, status);
            }
        }
        tfinal = (MPI_Wtime() - t0) / (2 * n_tests);
    }
    else tfinal = 0;

    delete ping_tags;
    delete pong_tags;
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
                MPI_Send(data, size, MPI_FLOAT, proc, ping_tag, MPI_COMM_WORLD);
            }
            for (int j = 0; j < n_msgs; j++)
            { 
                proc = procs[j];
                MPI_Recv(data, size, MPI_FLOAT, proc, pong_tag, MPI_COMM_WORLD, 
                        &status);
            }
        }
        else
        {
            for (int j = 0; j < n_msgs; j++)
            {
                proc = procs[j];
                MPI_Recv(data, size, MPI_FLOAT, proc, ping_tag, MPI_COMM_WORLD,
                        &status);
            }
            for (int j = 0; j < n_msgs; j++)
            {
                proc = procs[j];
                MPI_Send(data, size, MPI_FLOAT, proc, pong_tag, MPI_COMM_WORLD);
            }
        }
    }
    tfinal = (MPI_Wtime() - t0) / (2 * n_tests);



    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        if (master)
        {
            for (int j = 0; j < n_msgs; j++)
            {
                proc = procs[j];
                MPI_Send(data, size, MPI_FLOAT, proc, ping_tag, MPI_COMM_WORLD);
            }
            for (int j = 0; j < n_msgs; j++)
            { 
                proc = procs[j];
                MPI_Recv(data, size, MPI_FLOAT, proc, pong_tag, MPI_COMM_WORLD, 
                        &status);
            }
        }
        else
        {
            for (int j = 0; j < n_msgs; j++)
            {
                proc = procs[j];
                MPI_Recv(data, size, MPI_FLOAT, proc, ping_tag, MPI_COMM_WORLD,
                        &status);
            }
            for (int j = 0; j < n_msgs; j++)
            {
                proc = procs[j];
                MPI_Send(data, size, MPI_FLOAT, proc, pong_tag, MPI_COMM_WORLD);
            }
        }
    }
    tfinal = (MPI_Wtime() - t0) / (2 * n_tests);

    return tfinal;
}

        

