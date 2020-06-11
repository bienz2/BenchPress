#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <vector>

void timeThreeStep(int size, float* cpu_data, float* gpu_data, cudaStream_t& stream, int np, int gpu_rank)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int max_n = 14;
    int max_num = pow(2, max_n);
    MPI_Request* requests = new MPI_Request[max_num];

    float t0, tfinal;
    int n_tests = 1000;
    for (int n = 0; n < max_n; n++)
    {
        int n_msg = pow(2,n);

        if (size < n_msg ||  n_msg < np) 
        {
            if (rank == 0) printf("-1\t");
            continue;
        }

        int bytes = size * sizeof(float);
        int proc_n_msg = n_msg / np;
        int msg_size = size / n_msg;

        if (proc_n_msg > 1000 || bytes > 1000000) n_tests = 100;
        if (proc_n_msg > 10000 || bytes > 10000000) n_tests = 10;

        int extra = n_msg % np;
        int intern = proc_n_msg;
        if (extra > gpu_rank) intern += 1;
        int intersize = intern * msg_size;

        int ping_test = 1234;
        int pong_test = 4321;
        int intrastep1 = 5678;
        int intrastep2 = 8765;

        cudaDeviceSynchronize();
        cudaStreamSynchronize(stream);
        if (gpu_rank < np)
        {
            for (int j = 0; j < n_tests; j++)
            {
                if (rank % 2 == 0)
                {
                    if (gpu_rank == 0)
                    {
                        cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
                        cudaStreamSynchronize(stream);
                        int ctr = intersize; 
                        for (int i = 1; i < np; i++)
                        {
                            int s = proc_n_msg;
                            if (extra > i) s += 1;
                            s *= msg_size;
                            MPI_Send(&(cpu_data[ctr]), s, MPI_FLOAT, rank + (2*i), intrastep1, MPI_COMM_WORLD);
                            ctr += s;
                        }
                    }
                    else
                        MPI_Recv(cpu_data, intersize, MPI_FLOAT, rank - (2*gpu_rank), intrastep1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    for (int i = 0; i < intern; i++)
                    {
                        MPI_Isend(&(cpu_data[i*msg_size]), msg_size, MPI_FLOAT, rank+1, ping_test, MPI_COMM_WORLD, &(requests[i]));
                    }
                    MPI_Waitall(intern, requests, MPI_STATUSES_IGNORE);

                    for (int i = 0; i < intern; i++)
                    {
                        MPI_Irecv(&(cpu_data[i*msg_size]), msg_size, MPI_FLOAT, rank+1, pong_test, MPI_COMM_WORLD, &(requests[i]));
                    }
                    MPI_Waitall(intern, requests, MPI_STATUSES_IGNORE);

                    if (gpu_rank == 0)
                    {
                        int ctr = intersize;
                        for (int i = 1; i < np; i++)
                        {
                            int s = proc_n_msg;
                            if (extra > i) s += 1;
                            s *= msg_size;
                            MPI_Recv(&(cpu_data[ctr]), s, MPI_FLOAT, rank + (2*i), intrastep2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            ctr += s;
                        }
                        cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
                        cudaStreamSynchronize(stream);
                    }
                    else
                        MPI_Send(cpu_data, intersize, MPI_FLOAT, rank - (2*gpu_rank), intrastep2, MPI_COMM_WORLD);
                }
                else
                {
                    for (int i = 0; i < intern; i++)
                    {
                        MPI_Irecv(&(cpu_data[i*msg_size]), msg_size, MPI_FLOAT, rank-1, ping_test, MPI_COMM_WORLD, &(requests[i]));
                    }
                    MPI_Waitall(intern, requests, MPI_STATUSES_IGNORE);

                    if (gpu_rank == 0)
                    {
                        int ctr = intersize;
                        for (int i = 1; i < np; i++)
                        {
                            int s = proc_n_msg;
                            if (extra > i) s += 1;
                            s *= msg_size;
                            MPI_Recv(&(cpu_data[ctr]), s, MPI_FLOAT, rank + (2*i), intrastep1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            ctr += s;
                        }
                        cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
                        cudaStreamSynchronize(stream);
                    }
                    else
                        MPI_Send(cpu_data, intersize, MPI_FLOAT, rank - (2*gpu_rank), intrastep1, MPI_COMM_WORLD);

                    if (gpu_rank == 0)
                    {
                        cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
                        cudaStreamSynchronize(stream);
                        int ctr = intersize; 
                        for (int i = 1; i < np; i++)
                        {
                            int s = proc_n_msg;
                            if (extra > i) s += 1;
                            s *= msg_size;
                            MPI_Send(&(cpu_data[ctr]), s, MPI_FLOAT, rank + (2*i), intrastep2, MPI_COMM_WORLD);
                            ctr += s;
                        }
                    }
                    else
                        MPI_Recv(cpu_data, intersize, MPI_FLOAT, rank - (2*gpu_rank), intrastep2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    for (int i = 0; i < intern; i++)
                    {
                        MPI_Isend(&(cpu_data[i*msg_size]), msg_size, MPI_FLOAT, rank-1, pong_test, MPI_COMM_WORLD, &(requests[i]));
                    }
                    MPI_Waitall(intern, requests, MPI_STATUSES_IGNORE);
                }
            }
        }

        cudaDeviceSynchronize();
        cudaStreamSynchronize(stream);
        MPI_Barrier(MPI_COMM_WORLD);

        if (gpu_rank < np)
        {
            t0 = MPI_Wtime();
            for (int j = 0; j < n_tests; j++)
            {
                if (rank % 2 == 0)
                {
                    if (gpu_rank == 0)
                    {
                        cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
                        cudaStreamSynchronize(stream);
                        int ctr = intersize; 
                        for (int i = 1; i < np; i++)
                        {
                            int s = proc_n_msg;
                            if (extra > i) s += 1;
                            s *= msg_size;
                            MPI_Send(&(cpu_data[ctr]), s, MPI_FLOAT, rank + (2*i), intrastep1, MPI_COMM_WORLD);
                            ctr += s;
                        }
                    }
                    else
                        MPI_Recv(cpu_data, intersize, MPI_FLOAT, rank - (2*gpu_rank), intrastep1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    for (int i = 0; i < intern; i++)
                    {
                        MPI_Isend(&(cpu_data[i*msg_size]), msg_size, MPI_FLOAT, rank+1, ping_test, MPI_COMM_WORLD, &(requests[i]));
                    }
                    MPI_Waitall(intern, requests, MPI_STATUSES_IGNORE);

                    for (int i = 0; i < intern; i++)
                    {
                        MPI_Irecv(&(cpu_data[i*msg_size]), msg_size, MPI_FLOAT, rank+1, pong_test, MPI_COMM_WORLD, &(requests[i]));
                    }
                    MPI_Waitall(intern, requests, MPI_STATUSES_IGNORE);

                    if (gpu_rank == 0)
                    {
                        int ctr = intersize;
                        for (int i = 1; i < np; i++)
                        {
                            int s = proc_n_msg;
                            if (extra > i) s += 1;
                            s *= msg_size;
                            MPI_Recv(&(cpu_data[ctr]), s, MPI_FLOAT, rank + (2*i), intrastep2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            ctr += s;
                        }
                        cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
                        cudaStreamSynchronize(stream);
                    }
                    else
                        MPI_Send(cpu_data, intersize, MPI_FLOAT, rank - (2*gpu_rank), intrastep2, MPI_COMM_WORLD);
                }
                else
                {
                    for (int i = 0; i < intern; i++)
                    {
                        MPI_Irecv(&(cpu_data[i*msg_size]), msg_size, MPI_FLOAT, rank-1, ping_test, MPI_COMM_WORLD, &(requests[i]));
                    }
                    MPI_Waitall(intern, requests, MPI_STATUSES_IGNORE);

                    if (gpu_rank == 0)
                    {
                        int ctr = intersize;
                        for (int i = 1; i < np; i++)
                        {
                            int s = proc_n_msg;
                            if (extra > i) s += 1;
                            s *= msg_size;
                            MPI_Recv(&(cpu_data[ctr]), s, MPI_FLOAT, rank + (2*i), intrastep1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            ctr += s;
                        }
                        cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
                        cudaStreamSynchronize(stream);
                    }
                    else
                        MPI_Send(cpu_data, intersize, MPI_FLOAT, rank - (2*gpu_rank), intrastep1, MPI_COMM_WORLD);

                    if (gpu_rank == 0)
                    {
                        cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
                        cudaStreamSynchronize(stream);
                        int ctr = intersize; 
                        for (int i = 1; i < np; i++)
                        {
                            int s = proc_n_msg;
                            if (extra > i) s += 1;
                            s *= msg_size;
                            MPI_Send(&(cpu_data[ctr]), s, MPI_FLOAT, rank + (2*i), intrastep2, MPI_COMM_WORLD);
                            ctr += s;
                        }
                    }
                    else
                        MPI_Recv(cpu_data, intersize, MPI_FLOAT, rank - (2*gpu_rank), intrastep2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    for (int i = 0; i < intern; i++)
                    {
                        MPI_Isend(&(cpu_data[i*msg_size]), msg_size, MPI_FLOAT, rank-1, pong_test, MPI_COMM_WORLD, &(requests[i]));
                    }
                    MPI_Waitall(intern, requests, MPI_STATUSES_IGNORE);
                }
            }
            tfinal = ((MPI_Wtime() - t0) / n_tests) * 1000;
        }
        else tfinal = 0;

        MPI_Reduce(&tfinal, &t0, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("%1.5e\t", t0);
    }
    if (rank == 0) printf("\n");

    delete[] requests;
}


void timeThreeStepAsync(int size, float* cpu_data, float* gpu_data, cudaStream_t& stream, int np, int gpu_rank)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    float t0, tfinal;
    int n_tests = 1000;

    int max_n = 14;
    int max_num = pow(2, max_n);
    std::vector<MPI_Request> requests(max_num);

    for (int n = 0; n < max_n; n++)
    {
        int n_msg = pow(2, n);

        if (size < n_msg|| n_msg < np) 
        {
            if (rank == 0) printf("-1\t");
            continue;
        }

        int proc_n_msg = n_msg / np;

        if (proc_n_msg > 1000 || size*sizeof(float) > 1000000) n_tests = 100;
        if (proc_n_msg > 10000 || size*sizeof(float) > 10000000) n_tests = 10;

        int extra = n_msg % np;
        if (extra > gpu_rank) proc_n_msg += 1;
        int msg_size = size / n_msg;
        int proc_size = proc_n_msg * msg_size;
        int bytes = proc_size * sizeof(float);


        cudaDeviceSynchronize();
        cudaStreamSynchronize(stream);
        if (gpu_rank < np)
        {
            for (int j = 0; j < n_tests; j++)
            {
                if (rank % 2 == 0)
                {
                    cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    for (int i = 0; i < proc_n_msg; i++)
                        MPI_Isend(&(cpu_data[i*msg_size]), msg_size, MPI_FLOAT, rank+1, 1234, MPI_COMM_WORLD, &(requests[i]));
                    MPI_Waitall(proc_n_msg, requests.data(), MPI_STATUSES_IGNORE);
                    for (int i = 0; i < proc_n_msg; i++)
                        MPI_Irecv(&(cpu_data[i*msg_size]), msg_size, MPI_FLOAT, rank+1, 4321, MPI_COMM_WORLD, &(requests[i]));
                    MPI_Waitall(proc_n_msg, requests.data(), MPI_STATUSES_IGNORE);
                    cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
                    cudaStreamSynchronize(stream);
                }
                else
                {
                    for (int i = 0; i < proc_n_msg; i++)
                        MPI_Irecv(&(cpu_data[i*msg_size]), msg_size, MPI_FLOAT, rank-1, 1234, MPI_COMM_WORLD, &(requests[i]));
                    MPI_Waitall(proc_n_msg, requests.data(), MPI_STATUSES_IGNORE);
                    cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
                    cudaStreamSynchronize(stream);
                    cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    for (int i = 0; i < proc_n_msg; i++)
                        MPI_Isend(&(cpu_data[i*msg_size]), msg_size, MPI_FLOAT, rank-1, 4321, MPI_COMM_WORLD, &(requests[i]));
                    MPI_Waitall(proc_n_msg, requests.data(), MPI_STATUSES_IGNORE);
                } 
            }
        }

        cudaDeviceSynchronize();
        cudaStreamSynchronize(stream);
        MPI_Barrier(MPI_COMM_WORLD);

        if (gpu_rank < np)
        {
            t0 = MPI_Wtime();
            for (int j = 0; j < n_tests; j++)
            {
                if (rank % 2 == 0)
                {
                    cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    for (int i = 0; i < proc_n_msg; i++)
                        MPI_Isend(&(cpu_data[i*msg_size]), msg_size, MPI_FLOAT, rank+1, 1234, MPI_COMM_WORLD, &(requests[i]));
                    MPI_Waitall(proc_n_msg, requests.data(), MPI_STATUSES_IGNORE);
                    for (int i = 0; i < proc_n_msg; i++)
                        MPI_Irecv(&(cpu_data[i*msg_size]), msg_size, MPI_FLOAT, rank+1, 4321, MPI_COMM_WORLD, &(requests[i]));
                    MPI_Waitall(proc_n_msg, requests.data(), MPI_STATUSES_IGNORE);
                    cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
                    cudaStreamSynchronize(stream);
                }
                else
                {
                    for (int i = 0; i < proc_n_msg; i++)
                        MPI_Irecv(&(cpu_data[i*msg_size]), msg_size, MPI_FLOAT, rank-1, 1234, MPI_COMM_WORLD, &(requests[i]));
                    MPI_Waitall(proc_n_msg, requests.data(), MPI_STATUSES_IGNORE);
                    cudaMemcpyAsync(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice, stream);
                    cudaStreamSynchronize(stream);
                    cudaMemcpyAsync(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    for (int i = 0; i < proc_n_msg; i++)
                        MPI_Isend(&(cpu_data[i*msg_size]), msg_size, MPI_FLOAT, rank-1, 4321, MPI_COMM_WORLD, &(requests[i]));
                    MPI_Waitall(proc_n_msg, requests.data(), MPI_STATUSES_IGNORE);
                } 
            }
            tfinal = ((MPI_Wtime() - t0) / n_tests) * 1000;
        }
        else tfinal = 0;

        MPI_Reduce(&tfinal, &t0, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("%1.5e\t", t0);
    }
    if (rank == 0) printf("\n");
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    int max_i = 20;
    int max_bytes = pow(2,max_i-1) * sizeof(float);
    int n_times = 2 * max_i * num_gpus;
    int size;
    std::vector<float> times(n_times);
    std::vector<float> max_times(n_times);
    float* cpu_data;
    float* gpu_data;
    cudaMallocHost((void**)&cpu_data, max_bytes);


    int node_rank = rank / 2;
    int ppn = 36;
    int ppg = ppn / num_gpus; 
    int gpu = node_rank / ppg;
    int gpu_rank = node_rank % ppg;
    cudaSetDevice(gpu);
    cudaMalloc((void**)&gpu_data, max_bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
//cudaStream_t stream = 0;

    if (rank == 0) printf("3Step:\n");
    for (int np = 1; np <= ppg; np++)
    {
        if (rank == 0) printf("%d Procs Per GPU:\n", np);

        for (int i = 0; i < max_i; i++)
        {
            size = pow(2, i);

            if (rank == 0) printf("%d:\t", size);
            timeThreeStep(size, cpu_data, gpu_data, stream, np, gpu_rank);
        }
        if (rank == 0) printf("\n");
    }
    if (rank == 0) printf("\n\n");
    if (rank == 0) printf("3Step Async:\n");
    for (int np = 1; np <= ppg; np++)
    {
        if (rank == 0) printf("%d Procs Per GPU:\n", np);

        for (int i = 0; i < max_i; i++)
        {
            size = pow(2, i);

            if (rank == 0) printf("%d:\t", size);
            timeThreeStepAsync(size, cpu_data, gpu_data, stream, np, gpu_rank);
        }
        if (rank == 0) printf("\n");
    }

    cudaFree(gpu_data);
    cudaStreamDestroy(stream);
    cudaFreeHost(cpu_data);


    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        printf("ERROR!\n");
        exit( -1 );
    }


    MPI_Finalize();
}
