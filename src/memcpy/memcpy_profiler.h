#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>

void profile_host_to_device(int max_pow = 24, int n_tests = 1000);
void profile_device_to_host(int max_i = 24, int n_tests = 1000);
void profile_device_to_device(int max_i = 24, int n_tests = 1000);

