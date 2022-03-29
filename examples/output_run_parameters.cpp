// Copyright (c) 2015-2017, Raptor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <stdlib.h>

// Include raptor
#include "heterobench.hpp"

// This is a basic use case.
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    print_program_stats();

    MPI_Finalize();

    return 0;
}

