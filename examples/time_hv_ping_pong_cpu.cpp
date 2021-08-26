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

    int max_i  = 24;
    int n_iter = 1000;
    int n_msgs = 5;

    if (argc > 1) max_i  = atoi(argv[1]);
    if (argc > 2) n_iter = atoi(argv[2]);
    if (argc > 3) n_msgs = atoi(argv[3]);

    profile_high_volume_ping_pong(max_i, n_iter, n_msgs);

    MPI_Finalize();

    return 0;
}

