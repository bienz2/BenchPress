#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>

#ifndef COLLECTIVE_PROFILER_HPP
#define COLLECTIVE_PROFILER_HPP

void allreduce_profile_cuda_aware(int max_i = 24);
void allreduce_profile_3step(int max_i = 24);
void allreduce_profile_3step_extra_msg(int max_i = 24);
void allreduce_profile_3step_dup_devptr(int max_i = 24);

#endif
