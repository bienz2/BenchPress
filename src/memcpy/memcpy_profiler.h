#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>

#ifndef MEMCPY_PROFILER_HPP
#define MEMCPY_PROFILER_HPP

void profile_host_to_device(int max_pow = 24, int n_tests = 1000);
void profile_device_to_host(int max_i = 24, int n_tests = 1000);
void profile_device_to_device(int max_i = 24, int n_tests = 1000);

void profile_device_to_host_mult(int max_i = 24, int n_tests = 1000);
void profile_host_to_device_mult(int max_i = 24, int n_tests = 1000);
void profile_device_to_device_mult(int max_i = 24, int n_tests = 1000);

#endif
