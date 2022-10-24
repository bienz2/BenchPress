#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>

#ifndef ALLREDUCE_PROFILER_HPP
#define ALLREDUCE_PROFILER_HPP

#ifdef GPU_AWARE
void allreduce_profile_gpu_aware(int max_i = 24);
#endif

void allreduce_profile_3step(int max_i = 24);
void allreduce_profile_3step_extra_msg(int max_i = 24);
void allreduce_profile_3step_dup_devptr(int max_i = 24);

#endif
