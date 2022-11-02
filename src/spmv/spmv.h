#ifndef SPMV_HPP
#define SPMV_HPP

#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>
#include <string.h>
#include <stdlib.h>

struct GPUMat
{
    int* idx1;
    int* idx2;
    double* vals;
    int n_rows, n_cols, nnz;

    GPUMat(int _n_rows, int _n_cols, int _nnz,
            std::vector<int>& _idx1, std::vector<int>& idx2,
            std::vector<double>& _vals)
    {
        n_rows = _n_rows;
        n_cols = _n_cols; 
        nnz = _nnz;
  
        cudaMalloc((void**)&idx1, (n_rows+1)*sizeof(int));
        cudaMalloc((void**)&idx2, (nnz)*sizeof(int));
        cudaMalloc((void**)&vals, (nnz)*sizeof(double));

        cudaMemcpy(idx1, _idx1.data(), (n_rows+1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(idx2, _idx2.data(), nnz*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(vals, _vals.data(), nnz*sizeof(double), cudaMemcpyHostToDevice);
    }

    ~GPUMat()
    {
        cudaFree(idx1);
        cudaFree(idx2);
        cudaFree(vals);
    }
};

struct CommData
{
    int n_msgs;
    int s_msgs;
    std::vector<int> procs;
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<double> buffer;
    std::vector<MPI_Request> requests;

    CommData(int _n, int _s, std::vector<int>& _procs, std::vector<int>& _indptr)
    {
        n_msgs = _n;
        s_msgs = _s;
        procs.resize(n_msgs);
        indptr.resize(n_msgs+1);
        buffer.resize(s_msgs);
        requests.resize(n_msgs);
        indptr[0] = 0;
        for (int i = 0; i < n_msgs; i++)
        {
            procs[i] = _procs[i];
            indptr[i] = _indptr[i];
        }
    }

    CommData(int _n, int _s, std::vector<int>& _procs,
            std::vector<int>& _indptr, std::vector<int>& _idx)
    {
        n_msgs = _n;
        s_msgs = _s;
        procs.resize(n_msgs);
        indptr.resize(n_msgs+1);
        indices.resize(s_msgs);
        buffer.resize(s_msgs);
        requests.resize(n_msgs);
        indptr[0] = 0;
        for (int i = 0; i < n_msgs; i++)
        {
            procs[i] = _procs[i];
            indptr[i] = _indptr[i];
        }
        for (int i = 0; i < s_msgs; i++)
        {
            indices[i] = _idx[i];
        }
    }

};

struct CommPkg
{
    CommData* send_data;
    CommData* recv_data;
    MPI_Comm mpi_comm;
 
    CommPkg(int n_sends, int s_sends, std::vector<int>& send_procs,
            std::vector<int>& send_ptr, std::vector<int>& send_idx,
            int n_recvs, std::vector<int>& recv_procs,
            std::vector<int>& recv_ptr, MPI_Comm _mpi_comm)
    {
        send_data = new CommData(n_sends, s_sends, send_procs, 
                send_ptr, send_idx);
        recv_data = new CommData(n_recvs, recv_procs, recv_ptr);
        mpi_comm = _mpi_comm;
    }

    ~CommPkg()
    {
        delete send_data;
        delete recv_data;
    }
}

struct NAPCommPkg
{
    CommPkg* local_L_par_comm;
    CommPkg* local_S_par_comm;
    CommPkg* local_R_par_comm;
    CommPkg* global_par_comm;

    NAPCommPkg(CommPkg* L, CommPkg* S, CommPkg* R, CommPkg* G)
    {
        local_L_par_comm = L;
        local_S_par_comm = S;
        local_R_par_comm = R;
        global_par_comm = G;
    }

    ~NAPCommPkg()
    {
        delete local_L_par_comm;
        delete local_S_par_comm;
        delete local_R_par_comm;
        delete global_par_comm;
    }
}

void cuda_aware_spmv(GPUMat* d_A_on, GPUMat* d_A_off, double* d_x, double* d_b,
        double* d_x_dist, CommPkg* comm_pkg, int* send_indices, double* sendbuf,
        cudaStream_t& stream, MPI_Comm comm);
void copy_to_cpu_spmv(GPUMat* d_A_on, GPUMat* d_A_off, double* x,
        double* x_dist, double* d_x, double* d_b, double* d_x_dist,
        CommPkg* comm_pkg, int* send_indices, double* sendbuf, double* cpubuf,
        cudaStream_t& stream, MPI_Comm comm);
void copy_nap_spmv(GPUMat* d_A_on, GPUMat* d_A_off, double* d_x,
        double* d_x_dist, double* d_b,
        int* d_nap_sendidx, double* d_nap_sendbuf, double* nap_sendbuf,
        int* d_nap_recvidx, double* d_nap_recvbuf, double* nap_recvbuf,
        NAPCommPkg* comm_pkg, cudaStream_t& stream, MPI_Comm node_gpu_comm);
void dup_nap_spmv(GPUMat* d_A_on, GPUMat* d_A_off, double* d_x,
        double* d_x_dist, double* d_b,
        int* d_dup_sendidx, double* d_dup_sendbuf, double* dup_sendbuf,
        int* d_dup_recvidx, double* d_dup_recvbuf, double* dup_recvbuf,
        NAPCommPkg* comm_pkg, cudaStream_t& stream, MPI_Comm node_gpu_comm);
#endif
