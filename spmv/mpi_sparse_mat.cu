#include <string.h>
#include <stdlib.h>
#include <vector>

#include "raptor.hpp"

#include <cuda.h>
#include <cusparse.h>

using namespace raptor;

struct GPUMat
{
    int* idx1;
    int* idx2;
    double* vals;
    int n_rows, n_cols, nnz;

    GPUMat(Matrix* A)
    {
        n_rows = A->n_rows;
        n_cols = A->n_cols;
        nnz = A->nnz;

        cudaMalloc((void**)&idx1, (n_rows+1)*sizeof(int));
        cudaMalloc((void**)&idx2, (nnz)*sizeof(int));
        cudaMalloc((void**)&vals, (nnz)*sizeof(double));
        cudaMemcpy(idx1, A->idx1.data(), (n_rows+1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(idx2, A->idx2.data(), nnz*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(vals, A->vals.data(), nnz*sizeof(double), cudaMemcpyHostToDevice);
    }

    ~GPUMat()
    {
        cudaFree(idx1);
        cudaFree(idx2);
        cudaFree(vals);
    }
};


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
void communicate_T(ParComm* A_comm, T* sendbuf, T* tmpbuf, T* recvbuf, MPI_Comm comm)
{
    int proc, start, end;
    int tag = 8492;
    MPI_Datatype type = get_datatype(sendbuf);

    for (int i = 0; i < A_comm->recv_data->num_msgs; i++)
    {
        proc = A_comm->recv_data->procs[i];
        start = A_comm->recv_data->indptr[i];
        end = A_comm->recv_data->indptr[i+1];
        MPI_Isend(&(sendbuf[start]), end - start, type, proc, tag,
                comm, &(A_comm->recv_data->requests[i]));
    }

    for (int i = 0; i < A_comm->send_data->num_msgs; i++)
    {
        proc = A_comm->send_data->procs[i];
        start = A_comm->send_data->indptr[i];
        end = A_comm->send_data->indptr[i+1];
        MPI_Irecv(&(tmpbuf[start]), end - start, type, proc, tag,
                comm, &(A_comm->send_data->requests[i]));
    }

    MPI_Waitall(A_comm->recv_data->num_msgs,
            A_comm->recv_data->requests.data(),
            MPI_STATUSES_IGNORE);

    MPI_Waitall(A_comm->send_data->num_msgs,
            A_comm->send_data->requests.data(),
            MPI_STATUSES_IGNORE);

    for (int i = 0; i < A_comm->send_data->size_msgs; i++)
    {
        int idx = A_comm->send_data->indices[i];
        recvbuf[idx] = tmpbuf[i];
    }
}

template <typename T>
void init_comm(ParComm* A_comm, T* sendbuf, T* recvbuf, MPI_Comm comm)
{
    int proc, start, end;
    int tag = 2948;
    MPI_Datatype type = get_datatype(sendbuf);

    for (int i = 0; i < A_comm->recv_data->num_msgs; i++)
    {
        proc = A_comm->recv_data->procs[i];
        start = A_comm->recv_data->indptr[i];
        end = A_comm->recv_data->indptr[i+1];
        MPI_Irecv(&(recvbuf[start]), end - start, type, proc, tag,
                comm, &(A_comm->recv_data->requests[i]));
    }

    for (int i = 0; i < A_comm->send_data->num_msgs; i++)
    {
        proc = A_comm->send_data->procs[i];
        start = A_comm->send_data->indptr[i];
        end = A_comm->send_data->indptr[i+1];
        MPI_Isend(&(sendbuf[start]), end - start, type, proc, tag,
                 comm, &(A_comm->send_data->requests[i]));
    }
}    

template <typename T>
void init_comm_idx(ParComm* A_comm, T* sendbuf, T* idxbuf, T* recvbuf, MPI_Comm comm)
{
    int proc, start, end;
    int tag = 2948;
    
    for (int i = 0; i < A_comm->recv_data->num_msgs; i++)
    {
        proc = A_comm->recv_data->procs[i];
        start = A_comm->recv_data->indptr[i];
        end = A_comm->recv_data->indptr[i+1];
        MPI_Irecv(&(recvbuf[start]), end - start, MPI_DOUBLE, proc, tag,
                comm, &(A_comm->recv_data->requests[i]));
    }

    for (int i = 0; i < A_comm->send_data->num_msgs; i++)
    {
        proc = A_comm->send_data->procs[i];
        start = A_comm->send_data->indptr[i];
        end = A_comm->send_data->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            idxbuf[j] = sendbuf[A_comm->send_data->indices[j]];
        }
        MPI_Isend(&(idxbuf[start]), end - start, MPI_DOUBLE, proc, tag,
                 comm, &(A_comm->send_data->requests[i]));
    }
}


void finalize_comm(ParComm* A_comm)
{
    if (A_comm->send_data->num_msgs)
        MPI_Waitall(A_comm->send_data->num_msgs, A_comm->send_data->requests.data(),
                MPI_STATUSES_IGNORE);

    if (A_comm->recv_data->num_msgs)
        MPI_Waitall(A_comm->recv_data->num_msgs, A_comm->recv_data->requests.data(),
                MPI_STATUSES_IGNORE);
}

template <typename T>
void communicate(ParComm* A_comm, T* sendbuf, T* recvbuf, MPI_Comm comm)
{
    init_comm(A_comm, sendbuf, recvbuf, comm);
    finalize_comm(A_comm);
}


template <typename T>
void communicate_idx(ParComm* A_comm, T* sendbuf, T* idxbuf, T* recvbuf, MPI_Comm comm)
{
    init_comm_idx(A_comm, sendbuf, idxbuf, recvbuf, comm);
    finalize_comm(A_comm);
}


void nap_communicate(TAPComm* tap_comm, double* sendbuf, double* recvbuf)
{
    int idx;
    NonContigData* local_R_recv = (NonContigData*) tap_comm->local_R_par_comm->recv_data;

    communicate(tap_comm->local_S_par_comm, sendbuf,
            tap_comm->local_S_par_comm->recv_data->buffer.data(), 
            tap_comm->local_S_par_comm->mpi_comm);
    communicate(tap_comm->global_par_comm,
            tap_comm->local_S_par_comm->recv_data->buffer.data(),
            tap_comm->global_par_comm->recv_data->buffer.data(),
            tap_comm->global_par_comm->mpi_comm);
    communicate(tap_comm->local_R_par_comm,
            tap_comm->global_par_comm->recv_data->buffer.data(),
            recvbuf,
            tap_comm->local_R_par_comm->mpi_comm);
}

void cuda_aware_spmv(GPUMat* d_A_on, GPUMat* d_A_off, double* d_x, double* d_b,
        double* d_x_dist, ParComm* A_comm, int* send_indices, double* sendbuf, 
        cudaStream_t& stream, MPI_Comm comm)
{
    int size_msgs = A_comm->send_data->size_msgs;

    // Initialize Communication
    if (size_msgs)
        BufferKernel<<<ceil(size_msgs / 256.0), 256>>>
                (d_x, sendbuf, send_indices, size_msgs);

    init_comm(A_comm, sendbuf, d_x_dist, comm);

    // SpMV with Local Data
    gpu_spmv(d_A_on, d_x, d_b, 0);

    finalize_comm(A_comm);

    gpu_spmv(d_A_off, d_x_dist, d_b, 1);
}

void copy_to_cpu_spmv(GPUMat* d_A_on, GPUMat* d_A_off, double* x,
        double* x_dist, double* d_x, double* d_b, double* d_x_dist, 
        ParComm* A_comm, int* send_indices, double* sendbuf, double* cpubuf, 
        cudaStream_t& stream, MPI_Comm comm)
{
    int size_msgs = A_comm->send_data->size_msgs;

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
    init_comm(A_comm, cpubuf, x_dist, comm);
    finalize_comm(A_comm);

    // Copy data to GPU in correct positions
    cudaMemcpyAsync(d_x_dist, x_dist, size_msgs*sizeof(double), 
            cudaMemcpyHostToDevice, stream);
    //cudaStreamSynchronize(stream);
    gpu_spmv(d_A_off, d_x_dist, d_b, 1, stream);
}


// TODO - update this method to avoid communicate_idx method
//      - original memcpy of size global sends (from all procs)
//      - final memcpy of size global recvs (to all procs)
//      - copy duplicates with memcpy and send duplicate data with local comm
void copy_nap_spmv(GPUMat* d_A_on, GPUMat* d_A_off, double* d_x,
        double* d_x_dist, double* d_b,
        int* d_nap_sendidx, double* d_nap_sendbuf, double* nap_sendbuf,
        int* d_nap_recvidx, double* d_nap_recvbuf, double* nap_recvbuf,
        TAPComm* tap_comm, cudaStream_t& stream, MPI_Comm node_gpu_comm)
{
    int size_msgs = tap_comm->local_S_par_comm->send_data->size_msgs;

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

    nap_communicate(tap_comm, nap_sendbuf, nap_recvbuf);

    size_msgs = tap_comm->local_R_par_comm->recv_data->size_msgs;
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
        TAPComm* tap_comm, cudaStream_t& stream, MPI_Comm node_gpu_comm)
{
    int gpu_rank, size_msgs;
    MPI_Comm_rank(node_gpu_comm, &gpu_rank);

    size_msgs = tap_comm->global_par_comm->send_data->size_msgs;
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
    init_comm(tap_comm->global_par_comm,
        dup_sendbuf,
        dup_recvbuf,
        tap_comm->global_par_comm->mpi_comm);
    finalize_comm(tap_comm->global_par_comm);

    size_msgs = tap_comm->local_R_par_comm->send_data->size_msgs;
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


void extra_comm(TAPComm* tap_comm)
{
    double* buf = NULL;
    nap_communicate(tap_comm, buf, buf);
}


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double t0, tfinal;
    int n_tests = 1000;

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
 
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    int node_rank, ppn;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &ppn);
    int num_nodes = num_procs / ppn;
    int procs_per_gpu = ppn / num_gpus;
    int gpu = node_rank / procs_per_gpu;
    int gpu_rank = node_rank % procs_per_gpu;

    cudaSetDevice(gpu);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    MPI_Comm gpu_comm;
    MPI_Comm_split(MPI_COMM_WORLD, gpu_rank, rank, &gpu_comm);

    MPI_Comm node_gpu_comm;
    MPI_Comm_split(node_comm, gpu, rank, &node_gpu_comm);

    ParCSRMatrix* A = NULL;
    int global_num_rows = 0;
    int global_num_cols = 0;
    int local_num_rows = 0;
    int local_num_cols = 0;
    long nnz = 0;
    long global_nnz;
    if (gpu_rank == 0)
    {
        A = readParMatrix(argv[1], -1, -1, -1, -1, gpu_comm);
        global_num_rows = A->global_num_rows;
        global_num_cols = A->global_num_cols;
        local_num_rows = A->local_num_rows;
        local_num_cols = A->on_proc_num_cols;
        nnz = A->on_proc->nnz + A->off_proc->nnz;
        int max_n;
        int global_n = 0;
        for (int i = 0; i < A->comm->send_data->num_msgs; i++)
        {
            if (A->comm->send_data->procs[i] / procs_per_gpu != rank / procs_per_gpu)
                global_n++;
        }
        MPI_Allreduce(&global_n, &max_n, 1, MPI_INT, MPI_MAX, gpu_comm);
        if (rank == 0) printf("Max N %d\n", max_n);
    }
    else 
    {
        A = new ParCSRMatrix();
    }

    MPI_Allreduce(MPI_IN_PLACE, &global_num_rows, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &global_num_cols, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    int first_local_row = 0;
    int first_local_col = 0;
    MPI_Exscan(&local_num_rows, &first_local_row, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Exscan(&local_num_cols, &first_local_col, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    Topology* topology = new Topology();
    MPI_Comm_free(&(topology->local_comm));
    MPI_Comm_split(node_comm, gpu, rank, &(topology->local_comm));
    MPI_Comm_size(topology->local_comm, &(topology->PPN));
    topology->num_nodes = num_procs / topology->PPN;
    if (num_procs % topology->PPN) topology->num_nodes++;
    MPI_Comm_free(&node_comm);
    Partition* nap_partition = new Partition(global_num_rows, global_num_cols, 
            local_num_rows, local_num_cols, first_local_row, first_local_col, topology);
    topology->num_shared--;
    TAPComm* tap_comm = new TAPComm(nap_partition, A->off_proc_column_map);
    nap_partition->num_shared--;

    int size_msgs = tap_comm->local_S_par_comm->recv_data->size_msgs;
    std::vector<int> recvbuf(size_msgs);
    communicate(tap_comm->local_S_par_comm,
        tap_comm->local_S_par_comm->send_data->indices.data(),
        recvbuf.data(),
        tap_comm->local_S_par_comm->mpi_comm);
    int *d_dup_sendidx;
    double *d_dup_sendbuf, *dup_sendbuf;
    size_msgs = tap_comm->global_par_comm->send_data->size_msgs;
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
    if (gpu_rank == 0)
    {
        tap_comm->local_S_par_comm->send_data->size_msgs = size_msgs;
        tap_comm->local_S_par_comm->send_data->indices.resize(size_msgs);
        tap_comm->local_S_par_comm->send_data->buffer.resize(size_msgs);
        tap_comm->local_S_par_comm->send_data->int_buffer.resize(size_msgs);
        int n_msgs = 0;
        for (int i = 0; i < procs_per_gpu; i++)
        {
            if (node_gpu_sizes[i])
            {
                tap_comm->local_S_par_comm->send_data->procs[n_msgs] = i;
                int start = node_gpu_displs[i];
                int end = node_gpu_displs[i+1];
                for (int j = start; j < end; j++)
                {
                    tap_comm->local_S_par_comm->send_data->indices[j] = node_dup_sendidx[j];
                }
                tap_comm->local_S_par_comm->send_data->indptr[n_msgs+1] = end;
                n_msgs++;
            }
        }
    }
    size_msgs = tap_comm->global_par_comm->send_data->size_msgs;
    tap_comm->local_S_par_comm->recv_data->size_msgs = size_msgs;
    tap_comm->local_S_par_comm->recv_data->buffer.resize(size_msgs);
    tap_comm->local_S_par_comm->recv_data->int_buffer.resize(size_msgs);
    if (tap_comm->local_S_par_comm->recv_data->num_msgs)
    {
        tap_comm->local_S_par_comm->recv_data->indptr[1] = 
                tap_comm->global_par_comm->send_data->size_msgs;
    }
    

    NonContigData* local_R_recv = (NonContigData*)tap_comm->local_R_par_comm->recv_data;
    std::vector<int> dup_recvidx(tap_comm->local_R_par_comm->send_data->size_msgs);
    std::vector<int> dup_tmpidx(tap_comm->local_R_par_comm->send_data->size_msgs);
    communicate_T(tap_comm->local_R_par_comm,
            local_R_recv->indices.data(),
            dup_recvidx.data(),
            dup_tmpidx.data(),
            tap_comm->local_R_par_comm->mpi_comm);
    size_msgs = tap_comm->local_R_par_comm->send_data->num_msgs;
    int *d_dup_recvidx;
    double *d_dup_recvbuf, *dup_recvbuf;
    if (size_msgs)
    {
        for (int i = 0; i < size_msgs; i++)
        {
            int idx = tap_comm->local_R_par_comm->send_data->indices[i];
            dup_recvidx[i] = dup_tmpidx[idx];
        }

        cudaMalloc((void**)&d_dup_recvidx, dup_recvidx.size()*sizeof(int));
        cudaMemcpy(d_dup_recvidx, dup_recvidx.data(), dup_recvidx.size()*sizeof(int),
                cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_dup_recvbuf, dup_recvidx.size()*sizeof(double));
        cudaMallocHost((void**)&dup_recvbuf, dup_recvidx.size()*sizeof(double));
    }


    int max_n;
    MPI_Reduce(&(tap_comm->global_par_comm->send_data->num_msgs), &max_n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Max TAP N %d\n", max_n);


    MPI_Allreduce(&nnz, &global_nnz, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) printf("Global NNZ %lu, NNZ/GPU %lu\n", global_nnz, 
            global_nnz / (num_gpus*num_nodes));
    
    int max_n;
    MPI_Allreduce(&(A->comm->send_data->num_msgs), &max_n, 1, MPI_INT,
            MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("Max N: %d\n", max_n);

    MPI_Allreduce(&(A->comm->send_data->size_msgs), &max_n, 1, MPI_INT,
            MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("Max S: %d\n", max_n);

    double *d_x, *d_x_dist, *d_b;
    cudaIpcMemHandle_t x_handle, x_dist_handle;
    if (gpu_rank == 0)
    {

        GPUMat* d_A_on = new GPUMat(A->on_proc);
        GPUMat* d_A_off = new GPUMat(A->off_proc);
    
        double *x, *x_dist, *b, *b_cuda_aware, *b_three_step, *b_nap, *b_dup;
        if (A->local_num_rows)
        {
            cudaMalloc((void**)&d_x, A->on_proc_num_cols*sizeof(double));
            cudaMalloc((void**)&d_b, A->local_num_rows*sizeof(double));

            cudaMallocHost((void**)&x, A->on_proc_num_cols*sizeof(double));
            cudaMallocHost((void**)&b, A->local_num_rows*sizeof(double));
            cudaMallocHost((void**)&b_cuda_aware, A->local_num_rows*sizeof(double));
            cudaMallocHost((void**)&b_three_step, A->local_num_rows*sizeof(double));
            cudaMallocHost((void**)&b_nap, A->local_num_rows*sizeof(double));
            cudaMallocHost((void**)&b_dup, A->local_num_rows*sizeof(double));
        }

        if (A->off_proc_num_cols)
        {
            cudaMalloc((void**)&d_x_dist, A->off_proc_num_cols*sizeof(double));
            cudaMallocHost((void**)&x_dist, A->off_proc_num_cols*sizeof(double));
        }

        if (A->local_num_rows)
        {
            srand(time(NULL)*rank);
            for (int i = 0; i < A->on_proc_num_cols; i++)
                x[i] = (double)(rand()) / RAND_MAX;
            cudaMemcpy(d_x, x, A->on_proc_num_cols*sizeof(double), cudaMemcpyHostToDevice);
        }

        int *d_sendidx;
        double *d_sendbuf, *sendbuf;

        int *d_nap_sendidx, *d_nap_recvidx;
        double *nap_sendbuf, *d_nap_sendbuf;
        double *nap_recvbuf, *d_nap_recvbuf;

        size_msgs = A->comm->send_data->size_msgs;
        if (size_msgs)
        {
            cudaMalloc((void**)&d_sendidx, size_msgs*sizeof(int));
            cudaMemcpy(d_sendidx, A->comm->send_data->indices.data(), 
                    size_msgs*sizeof(int), cudaMemcpyHostToDevice);
            cudaMalloc((void**)&d_sendbuf, size_msgs*sizeof(double));
            cudaMallocHost((void**)&sendbuf, size_msgs*sizeof(double));
        }


        size_msgs = tap_comm->local_S_par_comm->send_data->size_msgs;
        if (size_msgs)
        {
            cudaMalloc((void**)&d_nap_sendidx, size_msgs*sizeof(int));
            cudaMemcpy(d_nap_sendidx, 
                    tap_comm->local_S_par_comm->send_data->indices.data(),
                    size_msgs*sizeof(int), cudaMemcpyHostToDevice);
            cudaMalloc((void**)&d_nap_sendbuf, size_msgs*sizeof(double));
            cudaMallocHost((void**)&nap_sendbuf, size_msgs*sizeof(double));
        }

        NonContigData* local_R_recv = (NonContigData*)tap_comm->local_R_par_comm->recv_data;
        size_msgs = local_R_recv->size_msgs;
        if (size_msgs)
        {
            cudaMalloc((void**)&d_nap_recvidx, size_msgs*sizeof(int));
            cudaMemcpy(d_nap_recvidx, local_R_recv->indices.data(),
                    size_msgs*sizeof(int), cudaMemcpyHostToDevice);
            cudaMalloc((void**)&d_nap_recvbuf, size_msgs*sizeof(double));
            cudaMallocHost((void**)&nap_recvbuf, size_msgs*sizeof(double));
        }

        cudaError_t err;
        err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Rank %d, CudaError %s\n", rank, cudaGetErrorString(err));

        cudaDeviceSynchronize();
        cuda_aware_spmv(d_A_on, d_A_off, d_x, d_b, d_x_dist, A->comm, 
                d_sendidx, d_sendbuf, stream, gpu_comm);
        cudaMemcpy(b_cuda_aware, d_b, A->local_num_rows*sizeof(double), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();
        copy_to_cpu_spmv(d_A_on, d_A_off, x, x_dist, d_x, d_b, d_x_dist, A->comm, 
                d_sendidx, d_sendbuf, sendbuf, stream, gpu_comm);
        cudaMemcpy(b_three_step, d_b, A->local_num_rows*sizeof(double), cudaMemcpyDeviceToHost);

        for (int i = 0; i < A->local_num_rows; i++)
        {
            if (b_cuda_aware[i] != b_three_step[i])
            {
                printf("CA != C2CPU\n");
                break;
            }
        }

        cudaDeviceSynchronize();
        copy_nap_spmv(d_A_on, d_A_off, d_x, d_x_dist, d_b,
            d_nap_sendidx, d_nap_sendbuf, nap_sendbuf,
            d_nap_recvidx, d_nap_recvbuf, nap_recvbuf,
            tap_comm, stream, gpu_comm);
        cudaMemcpy(b_nap, d_b, A->local_num_rows*sizeof(double), cudaMemcpyDeviceToHost);
        for (int i = 0; i < A->local_num_rows; i++)
        {
            if (b_cuda_aware[i] != b_nap[i])
            {
                printf("CA != NAP (%e vs %e)\n", b_cuda_aware[i], b_nap[i]);
                break;
            }
        }

        cudaDeviceSynchronize();
        MPI_Barrier(gpu_comm);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_tests; i++)
        {
            cuda_aware_spmv(d_A_on, d_A_off, d_x, d_b, d_x_dist, A->comm, 
                    d_sendidx, d_sendbuf, stream, gpu_comm);
        }
        tfinal = (MPI_Wtime() - t0) / n_tests;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, gpu_comm);
        if (rank == 0) printf("Cuda Aware : %e\n", t0);

        cudaDeviceSynchronize();
        MPI_Barrier(gpu_comm);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_tests; i++)
        {
            copy_to_cpu_spmv(d_A_on, d_A_off, x, x_dist, d_x, d_b, d_x_dist, A->comm, 
                    d_sendidx, d_sendbuf, sendbuf, stream, gpu_comm);
        }
        tfinal = (MPI_Wtime() - t0) / n_tests;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, gpu_comm);
        if (rank == 0) printf("Copy To CPU : %e\n", t0);

        cudaDeviceSynchronize();
        MPI_Barrier(gpu_comm);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_tests; i++)
        {
            copy_nap_spmv(d_A_on, d_A_off, d_x, d_x_dist, d_b,
                d_nap_sendidx, d_nap_sendbuf, nap_sendbuf,
                d_nap_recvidx, d_nap_recvbuf, nap_recvbuf,
                tap_comm, stream, gpu_comm);
        }
        tfinal = (MPI_Wtime() - t0) / n_tests;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, gpu_comm);
        if (rank == 0) printf("Node Aware : %e\n", t0);

        cudaIpcGetMemHandle(&x_handle, (void*)d_x);
        cudaIpcGetMemHandle(&x_dist_handle, (void*)d_x_dist);

        MPI_Bcast(&x_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, node_gpu_comm);
        MPI_Bcast(&x_dist_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, node_gpu_comm);

        cudaDeviceSynchronize();
        dup_nap_spmv(d_A_on, d_A_off, d_x, d_x_dist, d_b,
            d_dup_sendidx, d_dup_sendbuf, dup_sendbuf,
            d_dup_recvidx, d_dup_recvbuf, dup_recvbuf,
            tap_comm, stream, node_gpu_comm);
        cudaMemcpy(b_dup, d_b, A->local_num_rows*sizeof(double), cudaMemcpyDeviceToHost);
        for (int i = 0; i < A->local_num_rows; i++)
        {
            if (b_cuda_aware[i] != b_dup[i])
            {
                printf("CA != DUP (%e vs %e)\n", b_cuda_aware[i], b_dup[i]);
                break;
            }
        }


        cudaDeviceSynchronize();
        MPI_Barrier(gpu_comm);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_tests; i++)
        {
            dup_nap_spmv(d_A_on, d_A_off, d_x, d_x_dist, d_b,
                d_dup_sendidx, d_dup_sendbuf, dup_sendbuf,
                d_dup_recvidx, d_dup_recvbuf, dup_recvbuf,
                tap_comm, stream, node_gpu_comm);
        }
        tfinal = (MPI_Wtime() - t0) / n_tests;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, gpu_comm);
        if (rank == 0) printf("Dup Devptr Node Aware : %e\n", t0);

        if (A->comm->send_data->size_msgs)
        {
            cudaFree(d_sendidx);
            cudaFree(d_sendbuf);
            cudaFreeHost(sendbuf);
        }

        if (tap_comm->local_S_par_comm->send_data->size_msgs)
        {
            cudaFree(d_nap_sendidx);
            cudaFree(d_nap_sendbuf);
            cudaFreeHost(nap_sendbuf);
        }
        if (tap_comm->local_R_par_comm->recv_data->size_msgs)
        {
            cudaFree(d_nap_recvidx);
            cudaFree(d_nap_recvbuf);
            cudaFreeHost(nap_recvbuf);
        }
        MPI_Barrier(node_gpu_comm);

        if (A->local_num_rows)
        {
            cudaFree(d_x);
            cudaFree(d_b);
            cudaFreeHost(x);
            cudaFreeHost(b);
            cudaFreeHost(b_cuda_aware);
            cudaFreeHost(b_three_step);
            cudaFreeHost(b_nap);
            cudaFreeHost(b_dup);
        }

        if (A->off_proc_num_cols)
        {
            cudaFreeHost(x_dist);
            cudaFree(d_x_dist);
        }

        delete d_A_on;
        delete d_A_off;
    }
    else
    {
        extra_comm(tap_comm);
        for (int i = 0; i < n_tests; i++)
        {
            extra_comm(tap_comm);
        }

        MPI_Bcast(&x_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, node_gpu_comm);
        MPI_Bcast(&x_dist_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, node_gpu_comm);

        cudaIpcOpenMemHandle((void**) &d_x, x_handle, cudaIpcMemLazyEnablePeerAccess);
        cudaIpcOpenMemHandle((void**) &d_x_dist, x_dist_handle, cudaIpcMemLazyEnablePeerAccess);

        dup_nap_spmv(NULL, NULL, d_x, d_x_dist, NULL,
            d_dup_sendidx, d_dup_sendbuf, dup_sendbuf,
            d_dup_recvidx, d_dup_recvbuf, dup_recvbuf,
            tap_comm, stream, node_gpu_comm);

        for (int i = 0; i < n_tests; i++)
        {
            dup_nap_spmv(NULL, NULL, d_x, d_x_dist, NULL,
                d_dup_sendidx, d_dup_sendbuf, dup_sendbuf,
                d_dup_recvidx, d_dup_recvbuf, dup_recvbuf,
                tap_comm, stream, node_gpu_comm);
        }

        cudaIpcCloseMemHandle(d_x);
        cudaIpcCloseMemHandle(d_x_dist);
        MPI_Barrier(node_gpu_comm);
    }
    
    
    if (tap_comm->local_S_par_comm->recv_data->size_msgs)
    {
        cudaFree(d_dup_sendidx);
        cudaFree(d_dup_sendbuf);
        cudaFreeHost(dup_sendbuf);
    }

    if (tap_comm->local_R_par_comm->send_data->num_msgs)
    {
        cudaFree(d_dup_recvidx);
        cudaFree(d_dup_recvbuf);
        cudaFreeHost(dup_recvbuf);
    }

    delete tap_comm;
    delete A;

    cudaStreamDestroy(stream);
    MPI_Comm_free(&gpu_comm);
    MPI_Comm_free(&node_gpu_comm);

    MPI_Finalize();
}

