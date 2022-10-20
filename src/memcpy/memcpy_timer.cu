#include "memcpy_timer.h"

double time_memcpy(int bytes, float* orig_x, float* dest_x,
        cudaMemcpyKind copy_kind, cudaStream_t stream, 
        int n_tests)
{
    double t0, tfinal;

    // Warm Up
    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        cudaMemcpyAsync(dest_x, orig_x, bytes, copy_kind, stream);
        cudaStreamSynchronize(stream);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;

    // Time Memcpy
    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        cudaMemcpyAsync(dest_x, orig_x, bytes, copy_kind, stream);
        cudaStreamSynchronize(stream);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;

    return tfinal;
}

double time_memcpy_peer(int bytes, float* orig_x, float* dest_x,
        int orig_gpu, int dest_gpu, cudaStream_t stream, 
        int n_tests)
{
    double t0, tfinal;
  
    // Warm Up
    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        cudaMemcpyAsync(dest_x, orig_x, bytes, 
                cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
        cudaMemcpyAsync(orig_x, dest_x, bytes,
                cudaMemcpyDeviceToDevice, stream);
    }
    tfinal = (MPI_Wtime() - t0) / (2 * n_tests);

    // Time Memcpy 
    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        cudaMemcpyAsync(dest_x, orig_x, bytes, 
                cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
        cudaMemcpyAsync(orig_x, dest_x, bytes,
                cudaMemcpyDeviceToDevice, stream);
    }
    tfinal = (MPI_Wtime() - t0) / (2 * n_tests);

    return tfinal;
}

