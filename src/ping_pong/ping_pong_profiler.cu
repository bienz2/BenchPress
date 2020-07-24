#include "ping_pong_profiler.h"
#include "ping_pong_timer.h"

void profile_ping_pong(int max_i, int n_tests)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    float* data;
    int max_bytes = pow(2, max_i - 1) * sizeof(float);
    int nt;
    double time, max_time;
    bool active;
    
    cudaMallocHost((void**)&data, max_bytes);
/*
    for (int rank0 = 0; rank0 < num_procs; rank0++)
    {
        for (int rank1 = rank0+1; rank1 < num_procs; rank1++)
        {
            active = (rank == rank0 || rank == rank1);
            nt = n_tests;
            if (rank == 0) printf("CPU %d and CPU %d:\t", rank0, rank1);
            for (int i = 0; i < max_i; i++)
            {
                if (i > 14) nt = n_tests / 10;
                if (i > 20) nt = n_tests / 100;
                time = time_ping_pong(active, rank0, rank1, data, pow(2,i), nt);
                MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0) printf("%e\t", max_time);
            }
        }
    }
 */
    cudaFreeHost(data);
}

void profile_ping_pong_gpu(int max_i)
{
    int rank, num_procs; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    float* data;
    int max_bytes = pow(2, max_i - 1) * sizeof(float);
    double time;
    bool active;

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
            MPI_INFO_NULL, &node_comm);
    int gpu;
    MPI_Comm_rank(node_comm, &gpu);
    MPI_Comm_free(&node_comm);

    cudaSetDevice(gpu);
    cudaMalloc((void**)&data, max_bytes);

    for (int rank0 = 0; rank0 < num_procs; rank0++)
    {
        for (int rank1 = 0; rank1 < num_procs; rank1++)
        {
            int n_tests = 1000;
            active = (rank == rank0 || rank == rank1);
            if (rank == 0) printf("GPU on rank %d and GPU on rank %d:\t", rank0, rank1);
            for (int i = 0; i < max_i; i++)
            {
                if (i > 14) n_tests = 100;
                if (i > 20) n_tests = 10;
                time = time_ping_pong(active, rank0, rank1, data, pow(2,i), n_tests);
                MPI_Reduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX,
                        0, MPI_COMM_WORLD);
                if (rank == 0) printf("%e\t", time);
            }
        }
    }

    cudaFree(data);
}

// TODO -- Assumes SMP ordering
void profile_max_rate(bool split_data, int max_i)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    float* data;
    int max_bytes = pow(2, max_i - 1) * sizeof(float);
    double time;
    int size, msg_size;
    bool active;

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
            MPI_INFO_NULL, &node_comm);
    int ppn, node_rank;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &ppn);
    MPI_Comm_free(&node_comm);

    cudaMallocHost((void**)&data, max_bytes);

    int partner;
    if (rank < ppn) partner = rank + ppn;
    else partner = rank - ppn;

    int n_tests = 1000;
    for (int i = 0; i < max_i; i++)
    {
        size = pow(2, i);
        msg_size = size;
        if (rank == 0) printf("Size %d\n", size);
        if (i > 14) n_tests = 100;
        if (i > 20) n_tests = 10;
        for (int np = 1; np <= ppn; np++)
        {
            if (split_data) msg_size = size / np;
            if (msg_size < 1) break;
            active = node_rank < np;
            time = time_ping_pong(active, rank, partner, data, 
                    msg_size, n_tests);  
            MPI_Reduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX,
                    0, MPI_COMM_WORLD);
            if (rank == 0) printf("%e\t", time);
        }
        if (rank == 0) printf("\n");
    }

    cudaFreeHost(data);
}


// TODO -- Assumes SMP Ordering 
//         AND procs on both sockets
void profile_max_rate_gpu(bool split_data, int max_i)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
 
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    float* data;
    int max_bytes = pow(2, max_i - 1) * sizeof(float);
    int n_tests, size, msg_size;
    double time;
    bool active;

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
            MPI_INFO_NULL, &node_comm);
    int ppn, node_rank;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &ppn);
    MPI_Comm_free(&node_comm);

    int procs_per_gpu = ppn / num_gpus;
    int gpu = num_gpus;
    if (node_rank % procs_per_gpu == 0)
    {
        gpu = node_rank / procs_per_gpu;
        cudaSetDevice(gpu);
        cudaMalloc((void**)&data, max_bytes);
    }

    int partner;
    if (rank < ppn) partner = rank + ppn;
    else partner = rank - ppn;

    n_tests = 1000;
    for (int i = 0; i < max_i; i++)
    {
        size = pow(2, i);
        msg_size = size;
        if (rank == 0) printf("Size %d\n", size);
        if (i > 14) n_tests = 100;
        if (i > 20) n_tests = 10;
        for (int np = 1; np <= num_gpus; np++)
        {
            if (split_data) msg_size = size / np;
            if (msg_size < 1) break;
            active = gpu < np;
            time = time_ping_pong(active, rank, partner, data,       
                    msg_size, n_tests);
            MPI_Reduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX,
                    0, MPI_COMM_WORLD);
            if (rank == 0) printf("%e\t", time);
        }
        if (rank == 0) printf("\n");
    }

    if (node_rank % procs_per_gpu == 0)
        cudaFreeHost(data);
}


// ASSUMES SMP ORDERING
void profile_ping_pong_mult(int max_i)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    float* data;
    int max_bytes = pow(2, max_i - 1) * sizeof(float);
    int n_msgs, n_tests, size;
    int* procs = NULL;
    double time;
    bool master;

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
            MPI_INFO_NULL, &node_comm);
    int ppn, node_rank;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &ppn);
    MPI_Comm_free(&node_comm);

    int procs_per_gpu = ppn / num_gpus;
    int gpu = num_gpus;
    master = false;

    // Rank 0 is master rank for GPU 0
    if (rank == 0) 
    {
	    gpu = 0;
	    cudaSetDevice(gpu);
	    master = true;
        n_msgs = num_gpus;
        procs = new int[n_msgs];
        for (int i = 0; i < n_msgs; i++)
            procs[i] = ppn + procs_per_gpu * i;
        cudaMalloc((void**)&data, max_bytes * n_msgs);
    }
    // Node 1 holds partner ranks, 1 per GPU
    if (node_rank % procs_per_gpu == 0 && rank / ppn == 1)
    {
        gpu = node_rank / procs_per_gpu;
        cudaSetDevice(gpu);
	    n_msgs = 1;
        procs = new int[1];
        procs[0] = 0;
        cudaMalloc((void**)&data, max_bytes * n_msgs);
    }

    n_tests = 1000;
    for (int i = 0; i < max_i; i++)
    {
        size = pow(2, i);
        if (rank == 0) printf("Size %d\n", size);
        if (i > 14) n_tests = 100;
        if (i > 20) n_tests = 10;
        time = time_ping_pong_mult(master, n_msgs, procs, 
                data, size, n_tests);
        MPI_Reduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, 0,
                MPI_COMM_WORLD);
        if (rank == 0) printf("%e\t", time);
    }
    if (rank == 0) printf("\n");



    if (n_msgs)
    {
        delete[] procs;
        cudaFree(data);
    }
}
