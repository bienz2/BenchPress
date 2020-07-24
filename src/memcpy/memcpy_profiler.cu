#include "memcpy_profiler.h"
#include "memcpy_timer.h"

void profile_host_to_device(int max_i, int n_tests)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    int max_bytes = pow(2,max_i-1) * sizeof(float);
    int bytes, nt;
    double time, max_time;
    float* cpu_data;
    float* gpu_data;
    cudaMallocHost((void**)&cpu_data, max_bytes);
    
    // Time HostToDevice Memcpy Async
    for (int proc = 0; proc < num_procs; proc++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        for (int gpu = 0; gpu < num_gpus; gpu++)
        {
            cudaSetDevice(gpu);
            cudaMalloc((void**)&gpu_data, max_bytes);
            cudaStream_t proc_stream;
            cudaStreamCreate(&proc_stream);

            nt = n_tests;
            if (rank == 0) printf("CPU %d to GPU %d:\t", proc, gpu);
            for (int i = 0; i < max_i; i++)
            {
                if (i > 20) nt = n_tests / 10;
                if (i > 25) nt = n_tests / 100;
                bytes = pow(2, i) * sizeof(float);
                if (rank == proc) time = time_memcpy(bytes, cpu_data, gpu_data, 
                        cudaMemcpyHostToDevice, proc_stream, nt);
                else time = 0;
                MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0) printf("%e\t", max_time);
            }
            if (rank == 0) printf("\n");

            cudaFree(gpu_data);
            cudaStreamDestroy(proc_stream);
        }
    }
    
    cudaFreeHost(cpu_data);
}

void profile_device_to_host(int max_i, int n_tests)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    int max_bytes = pow(2,max_i-1) * sizeof(float);
    int bytes, nt;
    double time, max_time;
    float* cpu_data;
    float* gpu_data;
    cudaMallocHost((void**)&cpu_data, max_bytes);

    // Time DeviceToHost Memcpy Async
    for (int proc = 0; proc < num_procs; proc++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        for (int gpu = 0; gpu < num_gpus; gpu++)
        {
            cudaSetDevice(gpu);
            cudaMalloc((void**)&gpu_data, max_bytes);
            cudaStream_t proc_stream;
            cudaStreamCreate(&proc_stream);

            nt = n_tests;
            if (rank == 0) printf("GPU %d to CPU %d:\t", gpu, proc);
            for (int i = 0; i < max_i; i++)
            {
                if (i > 20) nt = n_tests / 10;
                if (i > 25) nt = n_tests / 100;
                bytes = pow(2, i) * sizeof(float);
                if (rank == proc) time = time_memcpy(bytes, gpu_data, cpu_data, 
                        cudaMemcpyDeviceToHost, proc_stream, nt);
                else time = 0;
                MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0) printf("%e\t", max_time);
            }
            if (rank == 0) printf("\n");

            cudaFree(gpu_data);
            cudaStreamDestroy(proc_stream);
        }
    }

    cudaFreeHost(cpu_data);
}

void profile_device_to_device(int max_i, int n_tests)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    int max_bytes = pow(2,max_i-1) * sizeof(float);
    int bytes, nt;
    double time, max_time;
    float* gpu0_data;
    float* gpu1_data;
    
    for (int proc = 0; proc < num_procs; proc++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == proc)
        {
            for (int gpu0 = 0; gpu0 < num_gpus; gpu0++)
            {
                cudaSetDevice(gpu0);
                cudaMalloc((void**)&gpu0_data, max_bytes);
                cudaStream_t proc_stream;
                cudaStreamCreate(&proc_stream);
                for (int gpu1 = gpu0 + 1; gpu1 < num_gpus; gpu1++)
                {
                    cudaSetDevice(gpu1);
                    cudaMalloc((void**)&gpu1_data, max_bytes);

                    nt = n_tests;
                    if (rank == 0) printf("CPU %d, GPU %d <-> GPU %d:\t", proc, gpu0, gpu1);
                    for (int i = 0; i < max_i; i++)
                    {
                        if (i > 20) nt = n_tests / 10;
                        if (i > 25) nt = n_tests / 100;
                        bytes = pow(2, i) * sizeof(float);
                        if (rank == proc) time = time_memcpy_peer(bytes, gpu0_data,
                                gpu1_data, gpu0, gpu1, proc_stream, nt);
                        else time = 0;
                        MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX,
                                0, MPI_COMM_WORLD);
                        if (rank == 0) printf("%e\t", max_time);
                    }
                    printf("\n");
                    cudaFree(gpu1_data);
                }
                cudaFree(gpu0_data);
                cudaStreamDestroy(proc_stream);
            }
        }
    }
}

