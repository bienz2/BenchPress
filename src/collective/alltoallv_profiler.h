#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>

#ifndef ALLTOALLV_PROFILER_HPP
#define ALLTOALLV_PROFILER_HPP

#ifdef CUDA_AWARE
void alltoallv_profile_cuda_aware(int max_i = 24, bool imsg = false);
#endif

void alltoallv_profile_3step(int max_i = 24, bool imsg = false);
void alltoallv_profile_3step_extra_msg(int max_i = 24, bool imsg = false);
void alltoallv_profile_3step_dup_devptr(int max_i = 24, bool imsg = false);

#endif
