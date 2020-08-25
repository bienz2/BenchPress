// Copyright (c) 2015-2017, Raptor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <stdlib.h>

// Include raptor
#include "heterobench.hpp"

// This is a basic use case.
int main(int argc, char *argv[])
{
    // set rank and number of processors
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int max_i = 18;
    int n_iter = 1000;

    if (argc > 1) max_i = atoi(argv[1]);
    if (argc > 2) n_iter = atoi(argv[2]);

    profile_max_rate(false, max_i, n_iter);

    profile_max_rate_gpu(false, max_i, n_iter);

    MPI_Finalize();

    return 0;
}

