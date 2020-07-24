#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>

double time_memcpy(int bytes, float* orig_x, float* dest_x,
        cudaMemcpyKind copy_kind, cudaStream_t stream, 
        int n_tests = 1000);
double time_memcpy_peer(int bytes, float* orig_x, float* dest_x,
        int orig_gpu, int dest_gpu, cudaStream_t stream, 
        int n_tests = 1000);

