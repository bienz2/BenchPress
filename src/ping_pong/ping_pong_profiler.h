#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>

void profile_ping_pong(int max_i = 26, int n_tests = 1000);
void profile_ping_pong_gpu(int max_i = 26, int n_tests = 1000);
void profile_max_rate(bool split_data = false, int max_i = 26, int n_tests = 1000);
void profile_max_rate_gpu(bool split_data = false, int max_i = 26, int n_tests = 1000);
void profile_ping_pong_mult(int max_i = 24, int n_tests = 1000);
void profile_ping_pong_mult_gpu(int max_i = 24, int n_tests = 1000);
