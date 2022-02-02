# Example Programs : 
The programs in this folder test the performance of different communication and collective operations on heterogenenous architectures.  

## Collective Examples :  
The collective examples time the performance of collective operations, testing multiple strategies:
* CUDA-Aware MPI : the collective operation is passed data in GPU memory.  The MPI implementation decides how to transport data, often utilizing GPUDirect and having all inner-steps communicated between GPU memory.
* Copy-to-CPU : Copy all data to a single CPU per GPU, and perform collective with standard MPI call on CPU memory
* Copy-to-Many-CPUs : Copy a portion of the data to each available CPU and perform smaller collective operations between CPU cores.  There are two methods for copying to the CPU : 
    * Extra Message : copy all data to a single CPU core.  This CPU core then redistributes the data among all available CPU cores per GPU.
    * Duplicate Device Pointer : Send a copy of the device data pointer to each of the available CPU cores.  Each CPU core then pulls a portion of the data directly from the GPU.

### Allreduce Performance : time\_allreduce
This program returns the cost of performing an MPI\_Allreduce(...) with CUDA-Aware MPI, copying the data to a single CPU before an allreduce among CPU cores, and copying the data to all available CPU cores, both with the extra message and duplicate device pointer approaches.

### Alltoall Performance : time\_alltoall

### Alltoallv Performance : time\_alltoallv
This program returns the cost of performing an MPI\_Alltoallv(...) with CUDA-Aware MPI, Copy-to-CPU, and both Copy-to-Many-CPUs approaches.  Furthermore, this program also tests performing the MPI\_Alltoallv with MPI\_Isend and MPI\_Irecv calls, implemented manually.  When many CPU cores are used, each sends only a fraction of the messages.
