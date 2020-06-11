#include "mpi.h"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int num_procs, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    float* cpu_data;
    cudaMallocHost((void**)&cpu_data, sizeof(float));

    if (rank == 0) printf("Num Procs %d, NumGPUs %d\n", num_procs, num_gpus);

    MPI_Finalize();
}
