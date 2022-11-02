#include "spmv.h"

#include <cuda.h>
#include <cusparse.h>


__global__ void SpMVKernel(int n_rows, int* rowptr, int* col_idx, double* data,
        double* x, double* b)
{   
    int start, end, col;
    double val, sum;
    
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (i < n_rows)
    {   
        sum = 0;
        start = rowptr[i];
        end = rowptr[i+1];
        for (int j = start; j < end; j++)
        {   
            col = col_idx[j];
            val = data[j];
            
            sum += val * x[col];
        }
        b[i] += sum;
    }
}

__global__ void BufferKernel(double* orig_buf, double* sendbuf, int* send_indices, int n_copies)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < n_copies)
    {
        int idx = send_indices[i];
        sendbuf[i] = orig_buf[idx];
    }
}

__global__ void TAPBufferKernel(double* orig_buf, double* sendbuf, int* send_idx, int send_size)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    int idx;
    if (i < send_size)
    {
        idx = send_idx[i];
        sendbuf[i] = orig_buf[idx];
    }
}

__global__ void TAPRecvBufferKernel(double* orig_buf, double* recvbuf, int* recv_idx, int recv_size)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    int idx;
    if (i < recv_size)
    {
        idx = recv_idx[i];
        recvbuf[idx] = orig_buf[i];
    }
}

void gpu_spmv(GPUMat* d_A, double* d_x, double* d_b, double beta, cudaStream_t stream = 0)
{
    cusparseMatDescr_t descr_A;
    cusparseCreateMatDescr(&descr_A);
    cusparseSetMatType(descr_A, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_A, CUSPARSE_INDEX_BASE_ZERO);

    cusparseHandle_t cusparse_handle;
    cusparseCreate(&cusparse_handle);
    cusparseSetStream(cusparse_handle, stream);
    cusparseStatus_t status;

    double alpha = 1.0;

    status = cusparseDcsrmv(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_A->n_rows, d_A->n_cols, d_A->nnz, &alpha, descr_A, d_A->vals,
            d_A->idx1, d_A->idx2, d_x, &beta, d_b);
}

MPI_Datatype get_datatype(int* buf)
{
    return MPI_INT;
}

MPI_Datatype get_datatype(double* buf)
{
    return MPI_DOUBLE;
}

template <typename T>
void communicate_T(CommPkg* comm_pkg, T* sendbuf, T* tmpbuf,
        T* recvbuf, MPI_Comm comm)
{
    int proc, start, end;
    int tag = 8492;
    MPI_Datatype type = get_datatype(sendbuf);
    
    // Transpose communication : send recv_data
    for (int i = 0; i < comm_pkg->recv_data->n_msgs; i++)
    {
        proc = comm_pkg->recv_data->procs[i];
        start = comm_pkg->recv_data->indptr[i];
        end = comm_pkg->recv_data->indptr[i+1];
        MPI_Isend(&(sendbuf[start]), end - start, type, proc, tag,
                comm, &(comm_pkg->recv_data->requests[i]));
    }
    
    // Transpose communication : recv send_data
    for (int i = 0; i < comm_pkg->send_data->n_msgs; i++)
    {
        proc = comm_pkg->send_data->procs[i];
        start = comm_pkg->send_data->indptr[i];
        end = comm_pkg->send_data->indptr[i+1];
        MPI_Irecv(&(tmpbuf[start]), end - start, type, proc, tag,
                comm, &(comm_pkg->send_data->requests[i]));
    }
    
    MPI_Waitall(comm_pkg->recv_data->n_msgs,
            comm_pkg->recv_data->requests.data(),
            MPI_STATUSES_IGNORE);
    MPI_Waitall(comm_pkg->send_data->n_msgs,
            comm_pkg->send_data->requests.data(),
            MPI_STATUSES_IGNORE);
    
    for (int i = 0; i < comm_pkg->send_data->s_msgs; i++)
    {
        idx = comm_pkg->send_data->indices[i];
        recvbuf[idx] = tmpbuf[i];
    }
}


template <typename T>
void init_comm(CommPkg* comm_pkg, T* sendbuf, T* recvbuf, MPI_Comm comm)
{
    int proc, start, end;
    int tag = 2948;
    MPI_Datatype type = get_datatype(sendbuf);

    for (int i = 0; i < comm_pkg->recv_data->n_msgs; i++)
    {
        proc = comm_pkg->recv_data->procs[i];
        start = comm_pkg->recv_data->indptr[i];
        end = comm_pkg->recv_data->indptr[i+1];
        MPI_Irecv(&(recvbuf[start]), end - start, type, proc, tag,
                comm, &(comm_pkg->recv_data->requests[i]));
    }

    for (int i = 0; i < comm_pkg->send_data->n_msgs; i++)
    {
        proc = comm_pkg->send_data->procs[i];
        start = comm_pkg->send_data->indptr[i];
        end = comm_pkg->send_data->indptr[i+1];
        MPI_Isend(&(sendbuf[start]), end - start, type, proc, tag,
                 comm, &(comm_pkg->send_data->requests[i]));
    }
}


template <typename T>
void init_comm_idx(CommPkg* comm_pkg, T* sendbuf, T* idxbuf, T* recvbuf, MPI_Comm comm)
{   
    int proc, start, end;
    int tag = 2948;
    
    for (int i = 0; i < comm_pkg->recv_data->n_msgs; i++)
    {   
        proc = comm_pkg->recv_data->procs[i];
        start = comm_pkg->recv_data->indptr[i];
        end = comm_pkg->recv_data->indptr[i+1];
        MPI_Irecv(&(recvbuf[start]), end - start, MPI_DOUBLE, proc, tag,
                comm, &(comm_pkg->recv_data->requests[i]));
    }
    
    for (int i = 0; i < comm_pkg->send_data->n_msgs; i++)
    {   
        proc = comm_pkg->send_data->procs[i];
        start = comm_pkg->send_data->indptr[i];
        end = comm_pkg->send_data->indptr[i+1];
        for (int j = start; j < end; j++)
        {   
            idxbuf[j] = sendbuf[comm_pkg->send_data->indices[j]];
        }
        MPI_Isend(&(idxbuf[start]), end - start, MPI_DOUBLE, proc, tag,
                 comm, &(comm_pkg->send_data->requests[i]));
    }
}

void finalize_comm(CommPkg* comm_pkg)
{
    if (comm_pkg->send_data->n_msgs)
        MPI_Waitall(comm_pkg->send_data->n_msgs, 
                comm_pkg->send_data->requests.data(),
                MPI_STATUSES_IGNORE);

    if (comm_pkg->recv_data->n_msgs)
        MPI_Waitall(comm_pkg->recv_data->n_msgs, 
                comm_pkg->recv_data->requests.data(),
                MPI_STATUSES_IGNORE);
}

template <typename T>
void communicate(CommPkg* comm_pkg, T* sendbuf, T* recvbuf, MPI_Comm comm)
{
    init_comm(comm_pkg, sendbuf, recvbuf, comm);
    finalize_comm(comm_pkg);
}


template <typename T>
void communicate_idx(CommPkg* comm_pkg, T* sendbuf, T* idxbuf, T* recvbuf, MPI_Comm comm)
{
    init_comm_idx(comm_pkg, sendbuf, idxbuf, recvbuf, comm);
    finalize_comm(comm_pkg);
}

void nap_communicate(NAPCommPkg* comm_pkg, double* sendbuf, double* recvbuf)
{
    communicate(comm_pkg->local_S_par_comm, sendbuf,
            comm_pkg->local_S_par_comm->recv_data->buffer.data(),
            comm_pkg->local_S_par_comm->mpi_comm);
    communicate(comm_pkg->global_par_comm,
            comm_pkg->local_S_par_comm->recv_data->buffer.data(),
            comm_pkg->global_par_comm->recv_data->buffer.data(),
            comm_pkg->global_par_comm->mpi_comm);
    communicate(comm_pkg->local_R_par_comm,
            comm_pkg->global_par_comm->recv_data->buffer.data(),
            recvbuf,
            comm_pkg->local_R_par_comm->mpi_comm);
}

void cuda_aware_spmv(GPUMat* d_A_on, GPUMat* d_A_off, double* d_x, double* d_b,
        double* d_x_dist, CommPkg* comm_pkg, int* send_indices, double* sendbuf,
        cudaStream_t& stream, MPI_Comm comm)
{   
    int size_msgs = comm_pkg->send_data->size_msgs;
    
    // Initialize Communication
    if (size_msgs)
        BufferKernel<<<ceil(size_msgs / 256.0), 256>>>
                (d_x, sendbuf, send_indices, size_msgs);
    
    init_comm(comm_pkg, sendbuf, d_x_dist, comm);
    
    // SpMV with Local Data
    gpu_spmv(d_A_on, d_x, d_b, 0);
    
    finalize_comm(comm_pkg);
    
    gpu_spmv(d_A_off, d_x_dist, d_b, 1);
}

void copy_to_cpu_spmv(GPUMat* d_A_on, GPUMat* d_A_off, double* x,
        double* x_dist, double* d_x, double* d_b, double* d_x_dist,
        CommPkg* comm_pkg, int* send_indices, double* sendbuf, double* cpubuf,
        cudaStream_t& stream, MPI_Comm comm)
{
    int size_msgs = comm_pkg->send_data->size_msgs;

    // Copy data to CPU in correct positions
    if (size_msgs)
    {
        BufferKernel<<<ceil(size_msgs / 256.0), 256, 0, stream>>>
                (d_x, sendbuf, send_indices, size_msgs);
        cudaMemcpyAsync(cpubuf, sendbuf, size_msgs*sizeof(double),
                cudaMemcpyDeviceToHost, stream);
    }

    gpu_spmv(d_A_on, d_x, d_b, 0);
    // Initialize Communication
    if (size_msgs)
        cudaStreamSynchronize(stream);
    init_comm(comm_pkg, cpubuf, x_dist, comm);
    finalize_comm(comm_pkg);

    // Copy data to GPU in correct positions
    cudaMemcpyAsync(d_x_dist, x_dist, size_msgs*sizeof(double),
            cudaMemcpyHostToDevice, stream);
    //cudaStreamSynchronize(stream);
    gpu_spmv(d_A_off, d_x_dist, d_b, 1, stream);
}

void copy_nap_spmv(GPUMat* d_A_on, GPUMat* d_A_off, double* d_x,
        double* d_x_dist, double* d_b,
        int* d_nap_sendidx, double* d_nap_sendbuf, double* nap_sendbuf,
        int* d_nap_recvidx, double* d_nap_recvbuf, double* nap_recvbuf,
        NAPCommPkg* comm_pkg, cudaStream_t& stream, MPI_Comm node_gpu_comm)
{
    int size_msgs = comm_pkg->local_S_par_comm->send_data->size_msgs;

    // Copy data to CPU in correct positions
    if (size_msgs)
    {
        TAPBufferKernel<<<ceil(size_msgs/256.0), 256>>>(d_x, d_nap_sendbuf, d_nap_sendidx, size_msgs);
        cudaMemcpyAsync(nap_sendbuf, d_nap_sendbuf, size_msgs*sizeof(double),
            cudaMemcpyDeviceToHost, stream);
    }

    // SpMV with Local Data
    gpu_spmv(d_A_on, d_x, d_b, 0);
    if (size_msgs)
        cudaStreamSynchronize(stream);

    nap_communicate(comm_pkg, nap_sendbuf, nap_recvbuf);

    size_msgs = comm_pkg->local_R_par_comm->recv_data->size_msgs;
    if (size_msgs)
    {
        cudaMemcpyAsync(d_nap_recvbuf, nap_recvbuf, size_msgs*sizeof(double),
                cudaMemcpyHostToDevice, stream);
        TAPRecvBufferKernel<<<ceil(size_msgs/256.0), 256, 0, stream>>>
                (d_nap_recvbuf, d_x_dist, d_nap_recvidx, size_msgs);
    }
    //cudaStreamSynchronize(stream);
    gpu_spmv(d_A_off, d_x_dist, d_b, 1);
}

void dup_nap_spmv(GPUMat* d_A_on, GPUMat* d_A_off, double* d_x,
        double* d_x_dist, double* d_b,
        int* d_dup_sendidx, double* d_dup_sendbuf, double* dup_sendbuf,
        int* d_dup_recvidx, double* d_dup_recvbuf, double* dup_recvbuf,
        NAPCommPkg* comm_pkg, cudaStream_t& stream, MPI_Comm node_gpu_comm)
{
    int gpu_rank, size_msgs;
    MPI_Comm_rank(node_gpu_comm, &gpu_rank);

    size_msgs = comm_pkg->global_par_comm->send_data->size_msgs;
    if (size_msgs)
    {
        TAPBufferKernel<<<ceil(size_msgs/256.0), 256, 0, stream>>>
                (d_x, d_dup_sendbuf, d_dup_sendidx, size_msgs);
        cudaMemcpyAsync(dup_sendbuf, d_dup_sendbuf, size_msgs*sizeof(double),
                cudaMemcpyDeviceToHost, stream);
    }

    // SpMV with Local Data
    if (gpu_rank == 0)
    {
        gpu_spmv(d_A_on, d_x, d_b, 0);
    }

    if (size_msgs)
    {
        cudaStreamSynchronize(stream);
    }
    init_comm(comm_pkg->global_par_comm,
        dup_sendbuf,
        dup_recvbuf,
        comm_pkg->global_par_comm->mpi_comm);
    finalize_comm(comm_pkg->global_par_comm);


    size_msgs = comm_pkg->local_R_par_comm->send_data->size_msgs;
    if (size_msgs)
    {
        cudaMemcpyAsync(d_dup_recvbuf, dup_recvbuf, size_msgs*sizeof(double),
                cudaMemcpyHostToDevice, stream);
        TAPRecvBufferKernel<<<ceil(size_msgs/256.0), 256, 0, stream>>>
                (d_dup_recvbuf, d_x_dist, d_dup_recvidx, size_msgs);
        cudaStreamSynchronize(stream);
    }

    MPI_Barrier(node_gpu_comm);
    if (gpu_rank == 0)
    {
        gpu_spmv(d_A_off, d_x_dist, d_b, 1);
    }
}

void extra_comm(NAPCommPkg* comm_pkg)
{
    double* buf = NULL;
    nap_communicate(comm_pkg, buf, buf);
}
