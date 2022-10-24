#include "memcpy_timer.h"

double time_memcpy(int bytes, float* orig_x, float* dest_x,
        hipMemcpyKind copy_kind, hipStream_t stream, 
        int n_tests)
{
    double t0, tfinal;

    // Warm Up
    hipDeviceSynchronize();
    hipStreamSynchronize(stream);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        hipMemcpyAsync(dest_x, orig_x, bytes, copy_kind, stream);
        hipStreamSynchronize(stream);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;

    // Time Memcpy
    hipDeviceSynchronize();
    hipStreamSynchronize(stream);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        hipMemcpyAsync(dest_x, orig_x, bytes, copy_kind, stream);
        hipStreamSynchronize(stream);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;

    return tfinal;
}

double time_memcpy_peer(int bytes, float* orig_x, float* dest_x,
        int orig_gpu, int dest_gpu, hipStream_t stream, 
        int n_tests)
{
    double t0, tfinal;
  
    // Warm Up
    hipDeviceSynchronize();
    hipStreamSynchronize(stream);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        hipMemcpyAsync(dest_x, orig_x, bytes, 
                hipMemcpyDeviceToDevice, stream);
        hipStreamSynchronize(stream);
        hipMemcpyAsync(orig_x, dest_x, bytes,
                hipMemcpyDeviceToDevice, stream);
    }
    tfinal = (MPI_Wtime() - t0) / (2 * n_tests);

    // Time Memcpy 
    hipDeviceSynchronize();
    hipStreamSynchronize(stream);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        hipMemcpyAsync(dest_x, orig_x, bytes, 
                hipMemcpyDeviceToDevice, stream);
        hipStreamSynchronize(stream);
        hipMemcpyAsync(orig_x, dest_x, bytes,
                hipMemcpyDeviceToDevice, stream);
    }
    tfinal = (MPI_Wtime() - t0) / (2 * n_tests);

    return tfinal;
}

