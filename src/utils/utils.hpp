#ifndef UTILS_HPP
#define UTILS_HPP

#include "mpi.h"

#ifdef HIP
#include "utils_hip.hpp"
#else
#include "utils_cuda.hpp"
#endif

#endif
