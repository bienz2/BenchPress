#include "stats.h"

void print_program_stats()
{
    // set rank and number of processors
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    gpuGetDeviceCount(&num_gpus);

    int ppn, num_nodes, procs_per_gpu;
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
            MPI_INFO_NULL, &node_comm);
    MPI_Comm_size(node_comm, &ppn);
    MPI_Comm_free(&node_comm);
    procs_per_gpu = ppn / num_gpus;
    num_nodes = num_procs / ppn;

    if (rank == 0)
    {
        printf("This program has been run with the following parameters:\n");
        printf("Number of Processes : %d\n", num_procs);
        printf("Number of Nodes : %d\n", num_nodes);
        printf("Number of Processes Per Node : %d\n", ppn);
        printf("Number of GPUs Per Node : %d\n", num_gpus);
        printf("Number of Processes Available Per GPU : %d\n", procs_per_gpu);
    }
}
