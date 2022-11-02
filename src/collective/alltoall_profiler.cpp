#include "alltoall_profiler.hpp"
#include "alltoall_timer.hpp"

#ifdef GPU_AWARE
void alltoall_profile_gpu_aware(int max_i)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    gpuGetDeviceCount(&num_gpus);

    int max_size = pow(2, max_i-1);
    int max_bytes = max_size * num_procs * sizeof(double);
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

    gpuSetDevice(gpu);
    gpuMalloc((void**)&gpu_data, max_bytes);

    MPI_Comm gpu_comm;
    MPI_Comm_split(MPI_COMM_WORLD, gpu_rank, rank, &gpu_comm);

    // Time Cuda-Aware Alltoall
    if (gpu_rank == 0) // Only one proc per GPU
    {
        if (rank == 0) printf("Cuda-Aware Alltoall:\n");
        n_tests = 1000;
        for (int i = 0; i < max_i; i++)
        {
           if (i > 14) n_tests = 100;
           if (i > 20) n_tests = 10;
           size = pow(2, i);
           time = time_alltoall(size, gpu_data, gpu_comm, n_tests);
           MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, gpu_comm);
           if (rank == 0) printf("%e\t", max_time);
        }
        if (rank == 0) printf("\n\n");
    }

    gpuFree(gpu_data);
    MPI_Comm_free(&gpu_comm);

    gpuError err = gpuGetLastError();
    if ( gpuSuccess != err )
    {
        printf("ERROR!\n");
        exit( -1 );
    }
}
#endif

void alltoall_profile_3step(int max_i)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    gpuGetDeviceCount(&num_gpus);

    int max_size = pow(2, max_i-1);
    int max_bytes = max_size * num_procs * sizeof(double);
    float* cpu_data;
    float* gpu_data;
    double time, max_time;
    gpuMallocHost((void**)&cpu_data, max_bytes);

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

    gpuSetDevice(gpu);
    gpuMalloc((void**)&gpu_data, max_bytes);
    gpuStream_t stream;
    gpuStreamCreate(&stream);

    MPI_Comm gpu_comm;
    MPI_Comm_split(MPI_COMM_WORLD, gpu_rank, rank, &gpu_comm);

    // Time 3-Step Alltoall
    if (gpu_rank == 0)
    {
        if (rank == 0) printf("3-Step Alltoall:\n");
        n_tests = 1000;
        for (int i = 0; i < max_i; i++)
        {
            if (i > 14) n_tests = 100;
            if (i > 20) n_tests = 10;
            size = pow(2, i);
            time = time_alltoall_3step(size, cpu_data, gpu_data, stream, gpu_comm, n_tests);
            MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, gpu_comm);
            if (rank == 0) printf("%e\t", max_time);
        }
        if (rank == 0) printf("\n\n");
    }

    gpuFree(gpu_data);
    gpuStreamDestroy(stream);
    gpuFreeHost(cpu_data);
    MPI_Comm_free(&gpu_comm);

    gpuError err = gpuGetLastError();
    if ( gpuSuccess != err )
    {
        printf("ERROR!\n");
        exit( -1 );
    }
}

void alltoall_profile_3step_extra_msg(int max_i)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    gpuGetDeviceCount(&num_gpus);

    int max_size = pow(2, max_i-1);
    int max_bytes = max_size * num_procs * sizeof(double);
    float* cpu_data;
    float* gpu_data;

    gpuMallocHost((void**)&cpu_data, max_bytes);

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

    gpuSetDevice(gpu);
    gpuMalloc((void**)&gpu_data, max_bytes);
    gpuStream_t stream;
    gpuStreamCreate(&stream);

    MPI_Comm gpu_comm;
    MPI_Comm_split(MPI_COMM_WORLD, gpu_rank, rank, &gpu_comm);

    // Time 3-Step, Extra Msg
    if (rank == 0) printf("3-Step Alltoall, Extra Message:\n");
    n_tests = 1000;

    for (int i = 0; i < max_i; i++)
    {
        if (i > 14) n_tests = 100;
        if (i > 20) n_tests = 10;
        size = pow(2, i);
        time = time_alltoall_3step_msg(size, cpu_data, gpu_data, ppg, node_rank, stream,
               gpu_comm, n_tests);
        MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("%e\t", max_time);
    }
    if (rank == 0) printf("\n\n");


    gpuFree(gpu_data);
    gpuStreamDestroy(stream);
    gpuFreeHost(cpu_data);
    MPI_Comm_free(&gpu_comm);

    gpuError err = gpuGetLastError();
    if ( gpuSuccess != err )
    {
        printf("ERROR!\n");
        exit( -1 );
    }
}

void alltoall_profile_3step_dup_devptr(int max_i)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    gpuGetDeviceCount(&num_gpus);

    int max_size = pow(2, max_i-1);
    int max_bytes = max_size * num_procs * sizeof(double);
    float* cpu_data;
    float* gpu_data;

    gpuMallocHost((void**)&cpu_data, max_bytes);

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

    gpuSetDevice(gpu);
    gpuMalloc((void**)&gpu_data, max_bytes);
    gpuStream_t stream;
    gpuStreamCreate(&stream);

    MPI_Comm gpu_comm;
    MPI_Comm_split(MPI_COMM_WORLD, gpu_rank, rank, &gpu_comm);

    // Time 3-Step, Duplicate DevPtr
    if (rank == 0) printf("3-Step Alltoall, Duplicate DevPtr:\n");
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
        time = time_alltoall_3step(msg_size, cpu_data, gpu_data, stream, gpu_comm, n_tests);
        MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("%e\t", max_time);
    }
    if (rank == 0) printf("\n\n");

    gpuFree(gpu_data);
    gpuStreamDestroy(stream);
    gpuFreeHost(cpu_data);
    MPI_Comm_free(&gpu_comm);

    gpuError err = gpuGetLastError();
    if ( gpuSuccess != err )
    {
        printf("ERROR!\n");
        exit( -1 );
    }
}
