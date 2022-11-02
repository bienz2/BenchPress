// Copyright (c) 2015-2017, Raptor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <stdlib.h>

// Include raptor
#include "benchpress.hpp"

// This is a basic use case.
int main(int argc, char *argv[])
{
    // set rank and number of processors
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Will test alltoallv with data size of 2^0 to 2^{max_i}
    // If 2^16 is too large, pass a command line argument for your selected max_i
    int max_i = 16;
    if (argc > 1) max_i = atoi(argv[1]);

    /* Timing Standard MPI\_Alltoallv() calls, with whatever implementation is included in MPI */

    // Time the cost of MPI\_Alltoallv(...) call on GPU Memory
#ifdef CUDA_AWARE
    alltoallv_profile_cuda_aware(max_i, false);
#endif

    // Time the cost of copying data to CPU and calling MPI\_Alltoallv(...) on CPU Memory
    // Using 1 CPU core per GPU
    alltoallv_profile_3step(max_i, false);

    // Time the cost of copying data to CPU and calling MPI\_Alltoallv(...) on CPU Memory
    // Copy data from GPU to 1 CPU core
    // After CPU core receives data, it redistributes data across all available CPU cores
    // 10 CPU cores per GPU on Lassen, 6 on Summit
    alltoallv_profile_3step_extra_msg(max_i, false);

    // Time the cost of copying data to CPU and calling MPI\_Alltoallv(...) on CPU Memory
    // Uses duplicate device pointer to copy a portion of the messages to each available CPU core
    // 10 CPU cores per GPU on Lassen, 6 on Summit
    alltoallv_profile_3step_dup_devptr(max_i, false);





    /* Timing send/recv implementation of MPI\_Alltoallv(), done by hand
     * This implementation initializes MPI\_Isend and MPI\_Irecv with all pairs of processes
     * It then calls MPI\_Waitall for both the MPI\_Isend and MPI\_Irecv requests
     */

    // Time the cost of MPI\_Isend, MPI\_Irecv, and MPI\_Waitall with all process pairs on GPU Memory
#ifdef CUDA_AWARE
    alltoallv_profile_cuda_aware(max_i, true);
#endif

    // Time the cost of copying data to the CPU and
    // MPI\_Isend, MPI\_Irecv, and MPI\_Waitall with all process pairs on GPU Memory
    // Using 1 CPU core per GPU
    alltoallv_profile_3step(max_i, true);

    // Time the cost of copying data to the CPU and
    // MPI\_Isend, MPI\_Irecv, and MPI\_Waitall with all process pairs on GPU Memory
    // After CPU core receives data, it redistributes data across all available CPU cores
    // 10 CPU cores per GPU on Lassen, 6 on Summit
    alltoallv_profile_3step_extra_msg(max_i, true);

    // Time the cost of copying data to the CPU and
    // MPI\_Isend, MPI\_Irecv, and MPI\_Waitall with all process pairs on GPU Memory
    // Uses duplicate device pointer to copy a portion of the messages to each available CPU core
    // 10 CPU cores per GPU on Lassen, 6 on Summit
    alltoallv_profile_3step_dup_devptr(max_i, true);

    MPI_Finalize();

    return 0;
}

