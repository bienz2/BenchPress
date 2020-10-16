#include "allreduce_profiler.h"
#include "allreduce_timer.h"

void allreduce_profile_cuda_aware(int max_i)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    int max_bytes = pow(2, max_i - 1) * sizeof(double);
    int n_tests, size;
    float* gpu_data;
    double time, max_time;

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
            &node_comm);
    int ppn, node_rank;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &ppn);
    MPI_Comm_free(&node_comm);

    int ppg = ppn / num_gpus;
    int gpu = node_rank / ppg;
    int gpu_rank = node_rank % ppg;

    cudaSetDevice(gpu);
    cudaMalloc((void**)&gpu_data, max_bytes);

    MPI_Comm gpu_comm;
    MPI_Comm_split(MPI_COMM_WORLD, gpu_rank, rank, &gpu_comm);

    // Time Cuda-Aware Allreduce
    if (gpu_rank == 0) // Only one proc per GPU
    {
        if (rank == 0) printf("Cuda-Aware Allreduce:\n");
        n_tests = 1000;
        for (int i = 0; i < max_i; i++)
        {
           if (i > 14) n_tests = 100;
           if (i > 20) n_tests = 10;
           size = pow(2, i);
           time = time_allreduce(size, gpu_data, gpu_comm, n_tests);
           MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, gpu_comm);
           if (rank == 0) printf("%e\t", max_time);
        }
        if (rank == 0) printf("\n\n");
    }

    cudaFree(gpu_data);
    MPI_Comm_free(&gpu_comm);

    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        printf("ERROR!\n");
        exit( -1 );
    }
}

void allreduce_profile_3step(int max_i)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    int max_bytes = pow(2, max_i - 1) * sizeof(double);
    float* cpu_data;
    float* gpu_data;
    double time, max_time;
    cudaMallocHost((void**)&cpu_data, max_bytes);

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
            &node_comm);
    int ppn, node_rank;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &ppn);
    MPI_Comm_free(&node_comm);

    int ppg = ppn / num_gpus;
    int gpu = node_rank / ppg;
    int gpu_rank = node_rank % ppg;
    int n_tests, size;

    cudaSetDevice(gpu);
    cudaMalloc((void**)&gpu_data, max_bytes);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    MPI_Comm gpu_comm;
    MPI_Comm_split(MPI_COMM_WORLD, gpu_rank, rank, &gpu_comm);

    
    // Time 3-Step Allreduce
    if (gpu_rank == 0)
    {
        if (rank == 0) printf("3-Step Allreduce:\n");
        n_tests = 1000;
        for (int i = 0; i < max_i; i++)
        {
            if (i > 14) n_tests = 100;
            if (i > 20) n_tests = 10;
            size = pow(2, i);
            time = time_allreduce_3step(size, cpu_data, gpu_data, stream, gpu_comm, n_tests);
            MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, gpu_comm);
            if (rank == 0) printf("%e\t", max_time);
        }
        if (rank == 0) printf("\n\n");
    }

    cudaFree(gpu_data);
    cudaStreamDestroy(stream);
    cudaFreeHost(cpu_data);
    MPI_Comm_free(&gpu_comm);

    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        printf("ERROR!\n");
        exit( -1 );
    }
}

void allreduce_profile_3step_extra_msg(int max_i)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    int max_bytes = pow(2, max_i - 1) * sizeof(double);
    float* cpu_data;
    float* gpu_data;

    cudaMallocHost((void**)&cpu_data, max_bytes);

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
            &node_comm);
    int ppn, node_rank;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &ppn);
    MPI_Comm_free(&node_comm);

    int ppg = ppn / num_gpus;
    int gpu = node_rank / ppg;
    int gpu_rank = node_rank % ppg;
    int n_tests, size;
    double time, max_time;

    cudaSetDevice(gpu);
    cudaMalloc((void**)&gpu_data, max_bytes);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    MPI_Comm gpu_comm;
    MPI_Comm_split(MPI_COMM_WORLD, gpu_rank, rank, &gpu_comm);

    // Time 3-Step, Extra Msg
    if (rank == 0) printf("3-Step Allreduce, Extra Message:\n");
    n_tests = 1000;

    for (int i = 0; i < max_i; i++)
    {
        if (i > 14) n_tests = 100;
        if (i > 20) n_tests = 10;
        size = pow(2, i);
        time = time_allreduce_3step_msg(size, cpu_data, gpu_data, ppg, node_rank, stream,
               gpu_comm, n_tests);
        MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("%e\t", max_time);
    }
    if (rank == 0) printf("\n\n");


    cudaFree(gpu_data);
    cudaStreamDestroy(stream);
    cudaFreeHost(cpu_data);
    MPI_Comm_free(&gpu_comm);

    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        printf("ERROR!\n");
        exit( -1 );
    }
}

void allreduce_profile_3step_dup_devptr(int max_i)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    int max_bytes = pow(2, max_i - 1) * sizeof(double);
    float* cpu_data;
    float* gpu_data;

    cudaMallocHost((void**)&cpu_data, max_bytes);

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
            &node_comm);
    int ppn, node_rank;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &ppn);
    MPI_Comm_free(&node_comm);

    int ppg = ppn / num_gpus;
    int gpu = node_rank / ppg;
    int gpu_rank = node_rank % ppg;
    int n_tests, size, msg_size;
    double time, max_time;

    cudaSetDevice(gpu);
    cudaMalloc((void**)&gpu_data, max_bytes);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    MPI_Comm gpu_comm;
    MPI_Comm_split(MPI_COMM_WORLD, gpu_rank, rank, &gpu_comm);

    // Time 3-Step, Duplicate DevPtr
    if (rank == 0) printf("3-Step Allreduce, Duplicate DevPtr:\n");
    n_tests = 1000;
    for (int i = 0; i < max_i; i++)
    {
        if (i > 14) n_tests = 100;
        if (i > 20) n_tests = 10;
        size = pow(2, i);
        msg_size = size / ppg;
        if (msg_size < 1)
        {
           if (rank == 0) printf("-1\t");
           continue;
        }
        time = time_allreduce_3step(msg_size, cpu_data, gpu_data, stream, gpu_comm, n_tests);
        MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("%e\t", max_time);
    }
    if (rank == 0) printf("\n\n");

    cudaFree(gpu_data);
    cudaStreamDestroy(stream);
    cudaFreeHost(cpu_data);
    MPI_Comm_free(&gpu_comm);

    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        printf("ERROR!\n");
        exit( -1 );
    }
}
