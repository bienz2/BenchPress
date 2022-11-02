#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>

#ifndef MEMCPY_TIMER_HPP
#define MEMCPY_TIMER_HPP

double time_memcpy(int bytes, float* orig_x, float* dest_x,
        gpuMemcpyKind copy_kind, gpuStream_t stream, 
        int n_tests = 1000);
double time_memcpy_peer(int bytes, float* orig_x, float* dest_x,
        int orig_gpu, int dest_gpu, gpuStream_t stream, 
        int n_tests = 1000);

#endif
