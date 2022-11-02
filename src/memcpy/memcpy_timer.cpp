#include "memcpy_timer.h"

double time_memcpy(int bytes, float* orig_x, float* dest_x,
        gpuMemcpyKind copy_kind, gpuStream_t stream, 
        int n_tests)
{
    double t0, tfinal;

    // Warm Up
    gpuDeviceSynchronize();
    gpuStreamSynchronize(stream);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        gpuMemcpyAsync(dest_x, orig_x, bytes, copy_kind, stream);
        gpuStreamSynchronize(stream);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;

    // Time Memcpy
    gpuDeviceSynchronize();
    gpuStreamSynchronize(stream);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        gpuMemcpyAsync(dest_x, orig_x, bytes, copy_kind, stream);
        gpuStreamSynchronize(stream);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;

    return tfinal;
}

double time_memcpy_peer(int bytes, float* orig_x, float* dest_x,
        int orig_gpu, int dest_gpu, gpuStream_t stream, 
        int n_tests)
{
    double t0, tfinal;
  
    // Warm Up
    gpuDeviceSynchronize();
    gpuStreamSynchronize(stream);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        gpuMemcpyAsync(dest_x, orig_x, bytes, 
                gpuMemcpyDeviceToDevice, stream);
        gpuStreamSynchronize(stream);
        gpuMemcpyAsync(orig_x, dest_x, bytes,
                gpuMemcpyDeviceToDevice, stream);
    }
    tfinal = (MPI_Wtime() - t0) / (2 * n_tests);

    // Time Memcpy 
    gpuDeviceSynchronize();
    gpuStreamSynchronize(stream);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        gpuMemcpyAsync(dest_x, orig_x, bytes, 
                gpuMemcpyDeviceToDevice, stream);
        gpuStreamSynchronize(stream);
        gpuMemcpyAsync(orig_x, dest_x, bytes,
                gpuMemcpyDeviceToDevice, stream);
    }
    tfinal = (MPI_Wtime() - t0) / (2 * n_tests);

    return tfinal;
}

