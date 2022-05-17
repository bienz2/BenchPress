#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>

#ifndef ALLTOALL_PROFILER_HPP
#define ALLTOALL_PROFILER_HPP

#ifdef CUDA_AWARE
void alltoall_profile_cuda_aware(int max_i = 24);
#endif 

void alltoall_profile_3step(int max_i = 24);
void alltoall_profile_3step_extra_msg(int max_i = 24);
void alltoall_profile_3step_dup_devptr(int max_i = 24);

#endif
