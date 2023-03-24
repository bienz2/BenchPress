#include "spmv_profiler.h"

void profile_cuda_aware_spmv(GPUMat* d_A_on, GPUMat* d_A_off, double* vector, 
        CommPkg* comm_pkg, cudaStream_t& stream, MPI_Comm gpu_comm, int n_tests)
{
    double *x, *d_x, *d_b, *d_x_dist, *d_sendbuf;
    int *d_sendidx, size_msgs, n_rows, n_on_cols, n_off_cols;

    size_msgs = comm_pkg->send_data->s_msgs;
    n_rows = d_A_on->n_rows;
    n_on_cols = d_A_on->n_cols;
    n_off_cols = d_A_off->n_cols;

    if (n_rows)
    {
        cudaMalloc((void**)&d_x, n_on_cols*sizeof(double));
        cudaMalloc((void**)&d_b, n_rows*sizeof(double));
            
        cudaMallocHost((void**)&x, n_on_cols*sizeof(double));

        for (int i = 0; i < n_on_cols; i++)
            x[i] = vector[i];
        cudaMemcpy(d_x, x, n_on_cols*sizeof(double), cudaMemcpyHostToDevice);
    }
    if (n_off_cols)
    {   
        cudaMalloc((void**)&d_x_dist, n_off_cols*sizeof(double));
    }
    if (size_msgs)
    {
        cudaMalloc((void**)&d_sendidx, size_msgs*sizeof(int));
        cudaMemcpy(d_sendidx, A->comm->send_data->indices.data(),
                size_msgs*sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_sendbuf, size_msgs*sizeof(double));
    }

    time_cuda_aware_spmv(d_A_on, d_A_off, d_x, d_b, d_x_dist, A->comm, 
        d_sendidx, d_sendbuf, stream, gpu_comm, n_tests);

    if (n_rows) 
    {
        cudaFree(d_x);
        cudaFree(d_b);
        cudaFreeHost(x);
    }
    if (n_off_cols)
    {
        cudaFree(d_x_dist);
    }
    if (size_msgs)
    {
        cudaFree(d_sendidx);
        cudaFree(d_sendbuf);
    }
}


void profile_copy_to_cpu_spmv(GPUMat* d_A_on, GPUMat* d_A_off, double* vector, 
        CommPkg* comm_pkg, cudaStream_t& stream, MPI_Comm gpu_comm, int n_tests)
{
    double *x, *x_dist, *d_x, *d_b, *d_x_dist;
    double *d_sendbuf, *sendbuf;
    int *d_sendidx, size_msgs, n_rows, n_on_cols, n_off_cols;

    size_msgs = comm_pkg->send_data->s_msgs;
    n_rows = d_A_on->n_rows;
    n_on_cols = d_A_on->n_cols;
    n_off_cols = d_A_off->n_cols;

    if (n_rows)
    {
        cudaMalloc((void**)&d_x, n_on_cols*sizeof(double));
        cudaMalloc((void**)&d_b, n_rows*sizeof(double));

        cudaMallocHost((void**)&x, n_on_cols*sizeof(double));

        for (int i = 0; i < n_on_cols; i++)
            x[i] = vector[i];
        cudaMemcpy(d_x, x, n_on_cols*sizeof(double), cudaMemcpyHostToDevice);
    }
    if (n_off_cols)
    {
        cudaMalloc((void**)&d_x_dist, n_off_cols*sizeof(double));
        cudaMallocHost((void**)&x_dist, n_off_cols*sizeof(double));
    }
    if (size_msgs)
    {
        cudaMalloc((void**)&d_sendidx, size_msgs*sizeof(int));
        cudaMemcpy(d_sendidx, A->comm->send_data->indices.data(),
                size_msgs*sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_sendbuf, size_msgs*sizeof(double));
        cudaMallocHost((void**)&sendbuf, size_msgs*sizeof(double));
    }

    time_copy_to_cpu_spmv(d_A_on, d_A_off, x, x_dist, d_x, d_b, d_x_dist,
            comm_pkg, d_sendidx, d_sendbuf, sendbuf, stream, gpu_comm, n_tests);

    if (n_rows)
    {
        cudaFree(d_x);
        cudaFree(d_b);
        cudaFreeHost(x);
    }
    if (n_off_cols)
    {
        cudaFree(d_x_dist);
        cudaFreeHost(x_dist);
    }
    if (size_msgs)
    {
        cudaFree(d_sendidx);
        cudaFree(d_sendbuf);
        cudaFreeHost(sendbuf);
    }
}



void profile_copy_nap_spmv(GPUMat* d_A_on, GPUMat* d_A_off, double* vector,
        NAPCommPkg* comm_pkg, cudaStream_t& stream, MPI_Comm gpu_comm, int n_tests)
{
    double *x, *d_x, *d_b, *d_x_dist;
    int size_msgs, n_rows, n_on_cols, n_off_cols;

    int *d_nap_sendidx, *d_nap_recvidx;
    double *nap_sendbuf, *d_nap_sendbuf;
    double *nap_recvbuf, *d_nap_recvbuf;

    n_rows = d_A_on->n_rows;
    n_on_cols = d_A_on->n_cols;
    n_off_cols = d_A_off->n_cols;

    if (n_rows)
    {
        cudaMalloc((void**)&d_x, n_on_cols*sizeof(double));
        cudaMalloc((void**)&d_b, n_rows*sizeof(double));

        cudaMallocHost((void**)&x, n_on_cols*sizeof(double));

        for (int i = 0; i < n_on_cols; i++)
            x[i] = vector[i];
        cudaMemcpy(d_x, x, n_on_cols*sizeof(double), cudaMemcpyHostToDevice);
    }
    if (n_off_cols)
    {
        cudaMalloc((void**)&d_x_dist, n_off_cols*sizeof(double));
    }
    size_msgs = comm_pkg->local_S_par_comm->send_data->s_msgs;
    if (size_msgs)
    {
        cudaMalloc((void**)&d_nap_sendidx, size_msgs*sizeof(int));
        cudaMemcpy(d_nap_sendidx,
                comm_pkg->local_S_par_comm->send_data->indices.data(),
                size_msgs*sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_nap_sendbuf, size_msgs*sizeof(double));
        cudaMallocHost((void**)&nap_sendbuf, size_msgs*sizeof(double));
    }
    size_msgs = comm_pkg->local_R_par_comm->recv_data->n_msgs;
    if (size_msgs)
    {
        cudaMalloc((void**)&d_nap_recvidx, size_msgs*sizeof(int));
        cudaMemcpy(d_nap_recvidx, 
                comm_pkg->local_R_par_comm->recv_data->indices.data(),
                size_msgs*sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_nap_recvbuf, size_msgs*sizeof(double));
        cudaMallocHost((void**)&nap_recvbuf, size_msgs*sizeof(double));
    }

    time_copy_nap_spmv(d_A_on, d_A_off, d_x, d_x_dist, d_b, 
            d_nap_sendidx, d_nap_sendbuf, nap_sendbuf, 
            d_nap_recvidx, d_nap_recvbuf, nap_recvbuf,
            comm_pkg, stream, gpu_comm, n_tests);


    if (n_rows)
    {
        cudaFree(d_x);
        cudaFree(d_b);
        cudaFreeHost(x);
    }
    if (n_off_cols)
    {
        cudaFree(d_x_dist);
    }
    if (comm_pkg->local_S_par_comm->send_data->size_msgs)
    {
        cudaFree(d_nap_sendidx);
        cudaFree(d_nap_sendbuf);
        cudaFreeHost(nap_sendbuf);
    }
    if (comm_pkg->local_R_par_comm->recv_data->s_msgs)
    {
        cudaFree(d_nap_recvidx);
        cudaFree(d_nap_recvbuf);
        cudaFreeHost(nap_recvbuf);
    }
}

void profile_dup_nap_spmv(GPUMat* d_A_on, GPUMat* d_A_off, double* vector,
        NAPCommPkg* comm_pkg, cudaStream_t& stream, MPI_Comm gpu_comm, int n_tests)
{
    int *d_dup_sendidx;
    double *d_dup_sendbuf, *dup_sendbuf;
    int size_msgs, n_rows, n_on_cols, n_off_cols;

    n_rows = d_A_on->n_rows;
    n_on_cols = d_A_on->n_cols;
    n_off_cols = d_A_off->n_cols;

    std::vector<int> recvbuf(comm_pkg->local_S_par_comm->recv_data->s_msgs);
    communicate(tap_comm->local_S_par_comm,
        tap_comm->local_S_par_comm->send_data->indices.data(),
        recvbuf.data(),
        tap_comm->local_S_par_comm->mpi_comm);
    size_msgs = comm_pkg->global_par_comm->send_data->s_msgs;
    std::vector<int> dup_sendidx(size_msgs);
    if (size_msgs)
    {
        for (int i = 0; i < size_msgs; i++)
        {
            int idx = tap_comm->global_par_comm->send_data->indices[i];
            dup_sendidx[i] = recvbuf[idx];
        }
        cudaMalloc((void**)&d_dup_sendidx, size_msgs*sizeof(int));
        cudaMemcpy(d_dup_sendidx, dup_sendidx.data(), size_msgs*sizeof(int),
                cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_dup_sendbuf, size_msgs*sizeof(double));
        cudaMallocHost((void**)&dup_sendbuf, size_msgs*sizeof(double));
    }

    // Allgather dup_sendidx on gpu_comm to have gpu_rank 0 hold indices 
    //     to send to each local proc
    size_msgs = tap_comm->global_par_comm->send_data->size_msgs;
    std::vector<int> node_dup_sendidx;
    std::vector<int> node_gpu_sizes(procs_per_gpu);
    std::vector<int> node_gpu_displs(procs_per_gpu+1);
    MPI_Gather(&size_msgs, 1, MPI_INT, node_gpu_sizes.data(), 1, MPI_INT,
            0, node_gpu_comm);
    if (gpu_rank == 0)
    {
        size_msgs = 0;
        node_gpu_displs[0] = 0;
        for (int i = 0; i < procs_per_gpu; i++)
        {
            size_msgs += node_gpu_sizes[i];
            node_gpu_displs[i+1] = node_gpu_displs[i] + node_gpu_sizes[i];
        }
        node_dup_sendidx.resize(size_msgs);
    }
    MPI_Gatherv(dup_sendidx.data(), dup_sendidx.size(), MPI_INT,
            node_dup_sendidx.data(), node_gpu_sizes.data(), node_gpu_displs.data(),
            MPI_INT, 0, node_gpu_comm);
}
