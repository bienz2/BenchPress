#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>

double time_ping_pong(bool active, int rank0, int rank1, float* data, 
        int size, int n_tests = 1000);
double time_ping_pong_3step(bool active, int rank0, int rank1, float* cpu_data, 
        float* gpu_data, int size, cudaStream_t stream, int n_tests = 1000);
double time_ping_pong_mult(bool master, int n_msgs, int* procs,
        float* data, int size, int n_tests = 1000);

        

