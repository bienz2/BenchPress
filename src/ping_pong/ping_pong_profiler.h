#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>

#ifndef PING_PONG_PROFILER_HPP
#define PING_PONG_PROFILER_HPP

void profile_ping_pong(int max_i = 26, int n_tests = 1000);
void profile_ping_pong_3step(int max_i = 26, int n_tests = 1000);
void profile_high_volume_ping_pong(int max_i = 26, int n_tests = 1000, int n_msgs = 5);
void profile_high_volume_ping_pong_3step(int max_i = 26, int n_tests = 1000, int n_msgs = 5);
void profile_max_rate(bool split_data = false, int max_i = 26, int n_tests = 1000);
void profile_ping_pong_mult(int max_i = 24, int n_tests = 1000, bool split_data = false);

#ifdef CUDA_AWARE
void profile_ping_pong_gpu(int max_i = 26, int n_tests = 1000);
void profile_high_volume_ping_pong_gpu(int max_i = 26, int n_tests = 1000, int n_msgs = 5);
void profile_max_rate_gpu(bool split_data = false, int max_i = 26, int n_tests = 1000);
void profile_ping_pong_mult_gpu(int max_i = 24, int n_tests = 1000, bool split_data = false);
#endif

#endif
