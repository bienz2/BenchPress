#include "alltoallv_profiler.h"
#include "alltoallv_timer.h"

// THIS METHOD CURRENTLY TESTS ALLTOALLV OPERATIONS WITH DATA ALL OF THE SAME SIZE

/*******************************************************************
 *** Method : alltoallv_profile_cuda_aware(...)
 ***
 ***    max_i : int
 ***        Will test sizes 2^{0} to 2^{max_i}
 ***    imsg : bool
 ***        If false, will call MPI\_Alltoallv(...)
 ***        If true, will call send_recv(...) in alltoallv_timer.h
 ***
 ***    This method profiles the cost of performing an MPI\_Alltoallv operation
 ***    with CUDA-Aware MPI.  This means the MPI\_Alltoallv method is performed
 ***    upon GPU memory.  The underlying MPI implementation determines how the 
 ***    data is transferred (e.g. GPUDirect?)
*******************************************************************/ 
void alltoallv_profile_cuda_aware(int max_i, bool imsg)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    int max_size = pow(2, max_i-1);
    int max_bytes = max_size * num_procs * sizeof(double);
    int n_tests, size;
    float* gpu_send_data;
    float* gpu_recv_data;
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
    cudaMalloc((void**)&gpu_send_data, max_bytes);
    cudaMalloc((void**)&gpu_recv_data, max_bytes);

    MPI_Comm gpu_comm;
    MPI_Comm_split(MPI_COMM_WORLD, gpu_rank, rank, &gpu_comm);

    // Time Cuda-Aware Alltoallv
    if (gpu_rank == 0) // Only one proc per GPU
    {
        if (rank == 0) printf("Cuda-Aware Alltoallv:\n");
        n_tests = 100;
        for (int i = 0; i < max_i; i++)
        {
           if (i > 14) n_tests = 100;
           if (i > 20) n_tests = 10;
           size = pow(2, i);
           if (imsg)
               time = time_alltoallv_imsg(size, gpu_send_data, gpu_recv_data, gpu_comm, n_tests);
           else
               time = time_alltoallv(size, gpu_send_data, gpu_recv_data, gpu_comm, n_tests);
           MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, gpu_comm);
           if (rank == 0) printf("%e\t", max_time);
        }
        if (rank == 0) printf("\n\n");
    }

    cudaFree(gpu_send_data);
    cudaFree(gpu_recv_data);
    MPI_Comm_free(&gpu_comm);

    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        printf("ERROR!\n");
        exit( -1 );
    }
}



/*******************************************************************
 *** Method : alltoallv_profile_3step(...)
 ***
 ***    max_i : int
 ***        Will test sizes 2^{0} to 2^{max_i}
 ***    imsg : bool
 ***        If false, will call MPI\_Alltoallv(...)
 ***        If true, will call send_recv(...) in alltoallv_timer.h
 ***
 ***    This method profiles the cost of performing an MPI\_Alltoallv operation
 ***    with the copy-to-CPU approach.  All data is first copied from each GPU
 ***    to a single CPU core.  The MPI\_Alltoallv operation is then performed
 ***    on this data, in CPU memory.
*******************************************************************/ 
void alltoallv_profile_3step(int max_i, bool imsg)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    int max_size = pow(2, max_i-1);
    int max_bytes = max_size * num_procs * sizeof(double);
    float* cpu_send_data;
    float* cpu_recv_data;
    float* gpu_data;
    double time, max_time;
    cudaMallocHost((void**)&cpu_send_data, max_bytes);
    cudaMallocHost((void**)&cpu_recv_data, max_bytes);

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

    // Time 3-Step Alltoallv
    if (gpu_rank == 0)
    {
        if (rank == 0) printf("3-Step Alltoallv:\n");
        n_tests = 100;
        for (int i = 0; i < max_i; i++)
        {
            if (i > 14) n_tests = 100;
            if (i > 20) n_tests = 10;
            size = pow(2, i);
            if (imsg)
                time = time_alltoallv_3step_imsg(size, cpu_send_data, cpu_recv_data,
                        gpu_data, stream, gpu_comm, n_tests);
            else
                time = time_alltoallv_3step(size, cpu_send_data, cpu_recv_data,
                        gpu_data, stream, gpu_comm, n_tests);
            MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, gpu_comm);
            if (rank == 0) printf("%e\t", max_time);
        }
        if (rank == 0) printf("\n\n");
    }

    cudaFree(gpu_data);
    cudaStreamDestroy(stream);
    cudaFreeHost(cpu_send_data);
    cudaFreeHost(cpu_recv_data);
    MPI_Comm_free(&gpu_comm);

    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        printf("ERROR!\n");
        exit( -1 );
    }
}


/*******************************************************************
 *** Method : alltoallv_profile_3step_extra_msg(...)
 ***
 ***    max_i : int
 ***        Will test sizes 2^{0} to 2^{max_i}
 ***    imsg : bool
 ***        If false, will call MPI\_Alltoallv(...)
 ***        If true, will call send_recv(...) in alltoallv_timer.h
 ***
 ***    This method profiles the cost of performing an MPI\_Alltoallv operation
 ***    with the copy-to-multiple-CPUs approach, using an extra message.
 ***    All data is first copied from each GPU to a single CPU core.  This CPU 
 ***    core then redistributes data among all available CPU cores per GPU, so that
 ***    each CPU core holds a fraction of the messages.  For instance, on Lassen 
 ***    there are 4 GPUs and 40 CPU cores per node.  The data is copied from each 
 ***    GPU to a single CPU core, which then redistributes to all other 9 avaialable
 ***    CPU cores per GPU so that each CPU core holds 1/10th of the messages.
 ***    The MPI\_Alltoallv operation is then performed on this data, in CPU memory.
*******************************************************************/ 
void alltoallv_profile_3step_extra_msg(int max_i, bool imsg)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    int max_size = pow(2, max_i-1);
    int max_bytes = max_size * num_procs * sizeof(double);
    float* cpu_send_data;
    float* cpu_recv_data;
    float* gpu_data;

    cudaMallocHost((void**)&cpu_send_data, max_bytes);
    cudaMallocHost((void**)&cpu_recv_data, max_bytes);

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
    if (rank == 0) printf("3-Step Alltoallv, Extra Message:\n");
    n_tests = 100;

    for (int i = 0; i < max_i; i++)
    {
        if (i > 14) n_tests = 100;
        if (i > 20) n_tests = 10;
        size = pow(2, i);
        if (imsg)
            time = time_alltoallv_3step_msg_imsg(size, cpu_send_data, cpu_recv_data, gpu_data, ppg, 
                   node_rank, stream, gpu_comm, n_tests);
        else
            time = time_alltoallv_3step_msg(size, cpu_send_data, cpu_recv_data, gpu_data, ppg, 
                   node_rank, stream, gpu_comm, n_tests);
        MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("%e\t", max_time);
    }
    if (rank == 0) printf("\n\n");


    cudaFree(gpu_data);
    cudaStreamDestroy(stream);
    cudaFreeHost(cpu_send_data);
    cudaFreeHost(cpu_recv_data);
    MPI_Comm_free(&gpu_comm);

    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        printf("ERROR!\n");
        exit( -1 );
    }
}


/*******************************************************************
 *** Method : alltoallv_profile_3step_dup_devptr(...)
 ***
 ***    max_i : int
 ***        Will test sizes 2^{0} to 2^{max_i}
 ***    imsg : bool
 ***        If false, will call MPI\_Alltoallv(...)
 ***        If true, will call send_recv(...) in alltoallv_timer.h
 ***
 ***    This method profiles the cost of performing an MPI\_Alltoallv operation
 ***    with the copy-to-multiple-CPUs approach, using a duplicate device pointer.
 ***    All available CPU cores per GPU (10 on Lassen) can access the pointer to 
 ***    the data on which the MPI\_Alltoallv must be performed.  Each CPU core
 ***    copies a portion of the data (1/10th of the messages, on Lassen) directly.
 ***    This means each CPU core calls cudaMemcpyAsync on a different offset 
 ***    for the same pointer to GPU memory.  The MPI\_Alltoallv operation is then
 ***    performed on this data, in CPU memory.
*******************************************************************/ 
void alltoallv_profile_3step_dup_devptr(int max_i, bool imsg)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    int max_size = pow(2, max_i-1);
    int max_bytes = max_size * num_procs * sizeof(double);
    float* cpu_send_data;
    float* cpu_recv_data;
    float* gpu_data;

    cudaMallocHost((void**)&cpu_send_data, max_bytes);
    cudaMallocHost((void**)&cpu_recv_data, max_bytes);

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

    // Time 3-Step, Duplicate DevPtr
    if (rank == 0) printf("3-Step Alltoallv, Duplicate DevPtr:\n");
    n_tests = 100;
    for (int i = 0; i < max_i; i++)
    {
        if (i > 14) n_tests = 100;
        if (i > 20) n_tests = 10;
        size = pow(2, i);
        if (imsg)
            time = time_alltoallv_3step_dup_imsg(size, cpu_send_data, cpu_recv_data, gpu_data, ppg, 
                   node_rank, stream, gpu_comm, n_tests);
        else
            time = time_alltoallv_3step_dup(size, cpu_send_data, cpu_recv_data, gpu_data, ppg, 
                   node_rank, stream, gpu_comm, n_tests);
        MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("%e\t", max_time);
    }
    if (rank == 0) printf("\n\n");

    cudaFree(gpu_data);
    cudaStreamDestroy(stream);
    cudaFreeHost(cpu_send_data);
    cudaFreeHost(cpu_recv_data);
    MPI_Comm_free(&gpu_comm);

    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        printf("ERROR!\n");
        exit( -1 );
    }
}
