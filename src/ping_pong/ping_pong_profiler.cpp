#include "ping_pong_profiler.h"
#include "ping_pong_timer.h"

void profile_ping_pong(int max_i, int n_tests)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
            MPI_INFO_NULL, &node_comm);
    int node_size, node_rank;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);
    MPI_Comm_free(&node_comm);

    float* data;
    int max_bytes = pow(2, max_i - 1) * sizeof(float);
    int nt;
    double time, max_time;
    bool active;
    
    gpuMallocHost((void**)&data, max_bytes);

    if (rank == 0) printf("Profiling Standard CPU Ping-Pongs:\n");
    for (int rank0 = 0; rank0 < node_size; rank0++)
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
            if (rank == 0) printf("\n");
        }
    }
    if (rank == 0) printf("\n\n");
 
    gpuFreeHost(data);
}

#ifdef GPU_AWARE
void profile_ping_pong_gpu(int max_i, int n_tests)
{
    int rank, num_procs; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    gpuGetDeviceCount(&num_gpus);

    float* data;
    int max_bytes = pow(2, max_i - 1) * sizeof(float);
    int nt;
    double time, max_time;
    bool active;

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
            MPI_INFO_NULL, &node_comm);
    int node_size, node_rank;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);
    MPI_Comm_free(&node_comm);
    int procs_per_gpu = node_size / num_gpus;
    int gpu = node_rank / procs_per_gpu;
    int total_num_gpus = num_procs / procs_per_gpu; 
    if (total_num_gpus == 1) 
    {
        printf("Only one GPU...\n");
        return;
    }

    gpuSetDevice(gpu);
    gpuMalloc((void**)&data, max_bytes);

    if (rank == 0) printf("Profiling GPU Ping-Pongs\n");
    for (int rank0 = 0; rank0 < node_size; rank0 += procs_per_gpu)
    {
        for (int rank1 = rank0+procs_per_gpu; rank1 < 2*node_size; rank1 += procs_per_gpu)
        {
            nt = n_tests;
            active = (rank == rank0 || rank == rank1);
            if (rank == 0) printf("GPU on rank %d and GPU on rank %d:\t", rank0, rank1);
            for (int i = 0; i < max_i; i++)
            {
                if (i > 14) nt = n_tests / 10;
                if (i > 20) nt = n_tests / 100;
                time = time_ping_pong(active, rank0, rank1, data, pow(2,i), nt);
                MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0) printf("%e\t", max_time);
            }
            if (rank == 0) printf("\n");
        }
    }
    if (rank == 0) printf("\n\n");

    gpuFree(data);
}
#endif

void profile_ping_pong_3step(int max_i, int n_tests)
{
    int rank, num_procs; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    gpuGetDeviceCount(&num_gpus);

    float* gpu_data;
    float* cpu_data;
    int max_bytes = pow(2, max_i - 1) * sizeof(float);
    int nt;
    double time, max_time;
    bool active;

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
            MPI_INFO_NULL, &node_comm);
    int node_size, node_rank;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);
    MPI_Comm_free(&node_comm);
    int procs_per_gpu = node_size / num_gpus;
    int gpu = node_rank / procs_per_gpu;
    int total_num_gpus = num_procs / procs_per_gpu; 
    if (total_num_gpus == 1) 
    {
        printf("Only one GPU...\n");
        return;
    }

    gpuMallocHost((void**)&cpu_data, max_bytes);
    gpuSetDevice(gpu);
    gpuMalloc((void**)&gpu_data, max_bytes);

    if (rank == 0) printf("Profiling GPU Ping-Pongs\n");
    for (int rank0 = 0; rank0 < node_size; rank0 += procs_per_gpu)
    {
        for (int rank1 = rank0+procs_per_gpu; rank1 < 2*node_size; rank1 += procs_per_gpu)
        {
            gpuStream_t proc_stream;
            gpuStreamCreate(&proc_stream);

            nt = n_tests;
            active = (rank == rank0 || rank == rank1);
            if (rank == 0) printf("GPU on rank %d and GPU on rank %d:\t", rank0, rank1);
            for (int i = 0; i < max_i; i++)
            {
                if (i > 14) nt = n_tests / 10;
                if (i > 20) nt = n_tests / 100;
                time = time_ping_pong_3step(active, rank0, rank1, cpu_data, gpu_data, pow(2,i), proc_stream, nt);
                MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0) printf("%e\t", max_time);
            }
            if (rank == 0) printf("\n");
            
            gpuStreamDestroy(proc_stream);
        }
    }
    if (rank == 0) printf("\n\n");

    gpuFree(gpu_data);
    gpuFreeHost(cpu_data);
}

void profile_high_volume_ping_pong(int max_i, int n_tests, int n_msgs)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
            MPI_INFO_NULL, &node_comm);
    int node_size, node_rank;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);
    MPI_Comm_free(&node_comm);

    float* data;
    int max_bytes = pow(2, max_i - 1) * sizeof(float);
    int nt;
    double time, max_time;
    bool active;
    
    gpuMallocHost((void**)&data, max_bytes);
    printf("after gpu malloc host\n");

    if (rank == 0) printf("Profiling High Volume CPU Ping-Pongs:\n");
    for (int rank0 = 0; rank0 < node_size; rank0++)
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
                time = time_high_volume_ping_pong(active, rank0, rank1, data, pow(2,i), nt, n_msgs);
                MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0) printf("%e\t", max_time);
            }
            if (rank == 0) printf("\n");
        }
    }
    if (rank == 0) printf("\n\n");
 
    gpuFreeHost(data);
}

#ifdef GPU_AWARE
void profile_high_volume_ping_pong_gpu(int max_i, int n_tests, int n_msgs)
{
    int rank, num_procs; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    gpuGetDeviceCount(&num_gpus);

    float* data;
    int max_bytes = pow(2, max_i - 1) * sizeof(float);
    int nt;
    double time, max_time;
    bool active;

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
            MPI_INFO_NULL, &node_comm);
    int node_size, node_rank;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);
    MPI_Comm_free(&node_comm);
    int procs_per_gpu = node_size / num_gpus;
    int gpu = node_rank / procs_per_gpu;
    int total_num_gpus = num_procs / procs_per_gpu; 
    if (total_num_gpus == 1) 
    {
        printf("Only one GPU...\n");
        return;
    }

    gpuSetDevice(gpu);
    gpuMalloc((void**)&data, max_bytes);

    if (rank == 0) printf("Profiling High Volume GPU Ping-Pongs\n");
    for (int rank0 = 0; rank0 < node_size; rank0 += procs_per_gpu)
    {
        for (int rank1 = rank0+procs_per_gpu; rank1 < 2*node_size; rank1 += procs_per_gpu)
        {
            nt = n_tests;
            active = (rank == rank0 || rank == rank1);
            if (rank == 0) printf("GPU on rank %d and GPU on rank %d:\t", rank0, rank1);
            for (int i = 0; i < max_i; i++)
            {
                if (i > 14) nt = n_tests / 10;
                if (i > 20) nt = n_tests / 100;
                time = time_high_volume_ping_pong(active, rank0, rank1, data, pow(2,i), nt, n_msgs);
                MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0) printf("%e\t", max_time);
            }
            if (rank == 0) printf("\n");
        }
    }
    if (rank == 0) printf("\n\n");

    gpuFree(data);
}
#endif


void profile_high_volume_ping_pong_3step(int max_i, int n_tests, int n_msgs)
{
    int rank, num_procs; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    gpuGetDeviceCount(&num_gpus);

    float* gpu_data;
    float* cpu_data;
    int max_bytes = pow(2, max_i - 1) * sizeof(float);
    int nt;
    double time, max_time;
    bool active;

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
            MPI_INFO_NULL, &node_comm);
    int node_size, node_rank;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);
    MPI_Comm_free(&node_comm);
    int procs_per_gpu = node_size / num_gpus;
    int gpu = node_rank / procs_per_gpu;
    int total_num_gpus = num_procs / procs_per_gpu; 
    if (total_num_gpus == 1) 
    {
        printf("Only one GPU...\n");
        return;
    }

    gpuMallocHost((void**)&cpu_data, max_bytes);
    gpuSetDevice(gpu);
    gpuMalloc((void**)&gpu_data, max_bytes);

    if (rank == 0) printf("Profiling High Volume GPU Ping-Pongs\n");
    for (int rank0 = 0; rank0 < node_size; rank0 += procs_per_gpu)
    {
        for (int rank1 = rank0+procs_per_gpu; rank1 < 2*node_size; rank1 += procs_per_gpu)
        {
            gpuStream_t proc_stream;
            gpuStreamCreate(&proc_stream);

            nt = n_tests;
            active = (rank == rank0 || rank == rank1);
            if (rank == 0) printf("GPU on rank %d and GPU on rank %d:\t", rank0, rank1);
            for (int i = 0; i < max_i; i++)
            {
                if (i > 14) nt = n_tests / 10;
                if (i > 20) nt = n_tests / 100;
                time = time_high_volume_ping_pong_3step(active, rank0, rank1, cpu_data, gpu_data,
                                                        pow(2,i), proc_stream, nt, n_msgs);
                MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0) printf("%e\t", max_time);
            }
            if (rank == 0) printf("\n");
            
            gpuStreamDestroy(proc_stream);
        }
    }
    if (rank == 0) printf("\n\n");

    gpuFree(gpu_data);
    gpuFreeHost(cpu_data);
}

// TODO -- Assumes SMP ordering
void profile_max_rate(bool split_data, int max_i, int n_tests)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    float* data;
    int max_bytes = pow(2, max_i - 1) * sizeof(float);
    double time, max_time;
    int size, msg_size, nt;
    bool active;

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
            MPI_INFO_NULL, &node_comm);
    int ppn, node_rank;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &ppn);
    MPI_Comm_free(&node_comm);
    int num_nodes = num_procs / ppn;
    int node = rank / ppn;

    if (num_nodes == 1)
    {
        if (rank == 0) printf("Only one node...\n");
        return;
    }

    gpuMallocHost((void**)&data, max_bytes);

    int master, partner;
    if ((rank / ppn) % 2 == 0)
    {
        if (node == num_nodes-1)
            node_rank = ppn;
        master = rank;
        partner = rank + ppn;
    }
    else
    {
        master = rank - ppn;
        partner = rank;
    }

    nt = n_tests;
    if (rank == 0) printf("Timing Max-Rate CPU Ping-Pongs\n");
    for (int i = 0; i < max_i; i++)
    {
        size = pow(2, i);
        msg_size = size;
        if (rank == 0) printf("Size %d\t", size);
        if (i > 14) nt = n_tests / 10;
        if (i > 20) nt = n_tests / 100;
        for (int np = 1; np <= ppn; np++)
        {
            if (split_data) msg_size = size / np;
            if (msg_size < 1) break;
            active = node_rank < np;
            time = time_ping_pong(active, master, partner, data, msg_size, nt);  
            MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("%e\t", max_time);
        }
        if (rank == 0) printf("\n");
    }
    if (rank == 0) printf("\n\n");

    gpuFreeHost(data);
}


// TODO -- Assumes SMP Ordering 
//         AND procs on both sockets
#ifdef GPU_AWARE
void profile_max_rate_gpu(bool split_data, int max_i, int n_tests)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
 
    int num_gpus;
    gpuGetDeviceCount(&num_gpus);

    float* data;
    int max_bytes = pow(2, max_i - 1) * sizeof(float);
    int nt, size, msg_size;
    double time, max_time;
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
    int num_nodes = num_procs / ppn;

    if (num_nodes == 1)
    {
        if (rank == 0) printf("Only one node...\n");
        return;
    }

    int node = rank / ppn;
    if (node_rank % procs_per_gpu == 0)
    {
        gpu = node_rank / procs_per_gpu;
        gpuSetDevice(gpu);
        gpuMalloc((void**)&data, max_bytes);
    }

    int master, partner;
    if ((rank / ppn) % 2 == 0)
    {
        master = rank;
        partner = rank + ppn;
        if (node == num_nodes-1) partner = master;
    }
    else
    {
        master = rank - ppn;
        partner = rank;
    }

    nt = n_tests;
    if (rank == 0) printf("Timing GPU Max-Rate Ping Pongs\n");
    for (int i = 0; i < max_i; i++)
    {
        size = pow(2, i);
        msg_size = size;
        if (rank == 0) printf("Size %d\t", size);
        if (i > 14) nt = n_tests / 10;
        if (i > 20) nt =  n_tests / 100;
        for (int np = 1; np <= num_gpus; np++)
        {
            if (split_data) msg_size = size / np;
            if (msg_size < 1) break;
            active = gpu < np && partner != master;
            time = time_ping_pong(active, master, partner, data, msg_size, nt);
            MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("%e\t", max_time);
        }
        if (rank == 0) printf("\n");
    }
    if (rank == 0) printf("\n\n");

    if (node_rank % procs_per_gpu == 0)
        gpuFree(data);
}
#endif 



void profile_ping_pong_mult(int max_i, int n_tests, bool split_data)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
            MPI_INFO_NULL, &node_comm);
    int ppn, node_rank;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &ppn);
    MPI_Comm_free(&node_comm);

    float* data = NULL;
    int max_bytes = pow(2, max_i - 1) * sizeof(float);
    int n_msgs, nt, size, s;
    int* procs = NULL;
    double time, max_time;
    bool master = false;
    int max_n = num_procs - ppn;

    int num_nodes = num_procs / ppn;
    if (num_nodes == 1)
    {
        if (rank == 0) printf("Only one node...\n");
        return;
    }

    gpuMallocHost((void**)&data, max_bytes);

    // Rank 0 is master
    if (rank == 0) 
    {
        master = true;
        n_msgs = num_procs - ppn;
        procs = new int[n_msgs];
        for (int i = 0; i < n_msgs; i++)
            procs[i] = ppn + i;
    }
    else if (rank >= ppn)
    {
        n_msgs = 1;
        procs = new int[n_msgs];
        procs[0] = 0;
    }
    else
    {
        n_msgs = 0;
        procs = NULL;
    }

    if (rank == 0) printf("Timing CPU Multiple Messages\n");
    nt = n_tests;
    int n_msg;
    for (int i = 0; i < max_i; i++)
    {
        size = pow(2, i);
        if (split_data && size < max_n) continue;

        if (rank == 0) printf("Size %d\t", size);
        if (i > 14) nt = n_tests / 10;
        if (i > 20) nt = n_tests / 100;
        for (int n = 0; n < max_n; n++)
        {
            if (split_data)
                s = size / (n+1);
            else s = size;
            if (rank == 0) 
                n_msg = n+1;
            else if (rank >= ppn && rank - ppn <= n)
                n_msg = 1;
            else
                n_msg = 0;
            time = time_ping_pong_mult(master, n_msg, procs, data, s, nt);
            MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("%e\t", max_time);
        }
        if (rank == 0) printf("\n");
    }
    if (rank == 0) printf("\n\n");

    if (n_msgs)
    {
        delete[] procs;
    }

    gpuFreeHost(data);
}

// ASSUMES SMP ORDERING
#ifdef GPU_AWARE
void profile_ping_pong_mult_gpu(int max_i, int n_tests, bool split_data)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    gpuGetDeviceCount(&num_gpus);

    float* data = NULL;
    int max_bytes = pow(2, max_i - 1) * sizeof(float);
    int n_msgs, nt, size, s;
    int* procs = NULL;
    double time, max_time;
    bool master;

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
            MPI_INFO_NULL, &node_comm);
    int ppn, node_rank;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &ppn);
    MPI_Comm_free(&node_comm);

    int procs_per_gpu = ppn / num_gpus;
    int num_nodes = num_procs / ppn;
    int gpu = num_gpus;
    master = false;

    if (num_nodes == 1)
    {
        if (rank == 0) printf("Only one node...\n");
        return;
    }

    gpuMalloc((void**)&data, max_bytes);

    // Rank 0 is master rank for GPU 0
    int max_n_msgs = num_gpus * (num_nodes-1);
    if (rank == 0) 
    {
        gpu = 0;
        gpuSetDevice(gpu);
        master = true;
        n_msgs = max_n_msgs;
        procs = new int[n_msgs];
        for (int i = 0; i < n_msgs; i++)
            procs[i] = ppn + (procs_per_gpu * i);
    }
    else if (node_rank % procs_per_gpu == 0 && rank >= ppn)
    {
        gpu = node_rank / procs_per_gpu;
        gpuSetDevice(gpu);
        n_msgs = 1;
        procs = new int[n_msgs];
        procs[0] = 0;
    }
    else
    {
        n_msgs = 0;
    }

    if (rank == 0) printf("Timing GPU Multiple Messages\n");
    nt = n_tests;
    int n_msg;
    for (int i = 0; i < max_i; i++)
    {
        size = pow(2, i);
        if (split_data && size < max_n_msgs) continue;
        if (rank == 0) printf("Size %d\t", size);
        if (i > 14) nt = n_tests / 10;
        if (i > 20) nt = n_tests / 100;
        for (int n = 0; n < max_n_msgs; n++)
        {
            if (split_data) 
                s = size / (n+1);
            else s = size;

            if (rank == 0) 
                n_msg = n+1;
            else if (rank >= ppn && node_rank % procs_per_gpu == 0 && 
                    ((rank-ppn) / procs_per_gpu) <= n) 
                n_msg = 1;
            else 
                n_msg = 0;
            time = time_ping_pong_mult(master, n_msg, procs, data, s, nt);
            MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0,
                    MPI_COMM_WORLD);
            if (rank == 0) printf("%e\t", max_time);
        }
        if (rank == 0) printf("\n");
    }
    if (rank == 0) printf("\n\n");



    if (n_msgs)
    {
        delete[] procs;
    }
    gpuFree(data);
}
#endif
