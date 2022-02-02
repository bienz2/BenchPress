#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>

#ifndef PING_PONG_TIMER_HPP
#define PING_PONG_TIMER_HPP

double time_ping_pong(bool active, int rank0, int rank1, float* data, 
        int size, int n_tests = 1000);
double time_high_volume_ping_pong(bool active, int rank0, int rank1, float* data, 
        int size, int n_tests = 1000, int n_msgs = 5);
double time_ping_pong_3step(bool active, int rank0, int rank1, float* cpu_data, 
        float* gpu_data, int size, cudaStream_t stream, int n_tests = 1000);
double time_high_volume_ping_pong_3step(bool active, int rank0, int rank1, float* cpu_data, 
        float* gpu_data, int size, cudaStream_t stream, int n_tests = 1000, int n_msgs = 5);
double time_ping_pong_mult(bool master, int n_msgs, int* procs,
        float* data, int size, int n_tests = 1000);

#endif        

