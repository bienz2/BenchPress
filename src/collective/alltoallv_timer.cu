#include "alltoallv_timer.h"

void send_recv(int size, int n_msgs, int* send_procs, int* recv_procs,
        MPI_Request* send_req, MPI_Request* recv_req, float* send_data, 
        float* recv_data, MPI_Comm& comm, int tag = 83205)
{
    for (int i = 0; i < n_msgs; i++)
    {
        MPI_Isend(&(send_data[i*size]), size, MPI_FLOAT, send_procs[i],
            tag, comm, &(send_req[i]));
        MPI_Irecv(&(recv_data[i*size]), size, MPI_FLOAT, recv_procs[i],
            tag, comm, &(recv_req[i]));
    }
    
    MPI_Waitall(n_msgs, send_req, MPI_STATUSES_IGNORE);
    MPI_Waitall(n_msgs, recv_req, MPI_STATUSES_IGNORE);
}

double time_alltoallv(int size, float* gpu_send_data, float* gpu_recv_data, MPI_Comm& group_comm,
        int n_tests)
{
    int  num_procs;
    MPI_Comm_size(group_comm, &num_procs);

    double t0, tfinal;

    std::vector<int> send_sizes(num_procs, size);
    std::vector<int> recv_sizes(num_procs, size);
    std::vector<int> send_displs(num_procs+1);
    std::vector<int> recv_displs(num_procs+1);
    send_displs[0] = 0;
    recv_displs[0] = 0;
    for (int i = 0; i < num_procs; i++)
    {
        send_displs[i+1] = send_displs[i] + send_sizes[i];
        recv_displs[i+1] = recv_displs[i] + recv_sizes[i];
    }

    // Warm Up
    cudaDeviceSynchronize();
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        MPI_Alltoallv(gpu_send_data, send_sizes.data(), send_displs.data(), MPI_FLOAT,
                gpu_recv_data, recv_sizes.data(), recv_displs.data(), MPI_FLOAT, group_comm);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;


    cudaDeviceSynchronize();
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        MPI_Alltoallv(gpu_send_data, send_sizes.data(), send_displs.data(), MPI_FLOAT,
                gpu_recv_data, recv_sizes.data(), recv_displs.data(), MPI_FLOAT, group_comm);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;

    return tfinal;
}


double time_alltoallv_imsg(int size, float* gpu_send_data, float* gpu_recv_data, MPI_Comm& group_comm,
        int n_tests)
{
    int  num_procs;
    MPI_Comm_size(group_comm, &num_procs);

    double t0, tfinal;

    std::vector<int> send_procs(num_procs);
    std::vector<int> recv_procs(num_procs);
    for (int i = 0; i < num_procs; i++)
    {
        send_procs[i] = i;
        recv_procs[i] = i;
    }
    std::vector<MPI_Request> send_req(num_procs);
    std::vector<MPI_Request> recv_req(num_procs);

    // Warm Up
    cudaDeviceSynchronize();
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        send_recv(size, num_procs, send_procs.data(), recv_procs.data(),
                send_req.data(), recv_req.data(), gpu_send_data, gpu_recv_data,
                group_comm);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;


    cudaDeviceSynchronize();
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        send_recv(size, num_procs, send_procs.data(), recv_procs.data(),
                send_req.data(), recv_req.data(), gpu_send_data, gpu_recv_data,
                group_comm);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;

    return tfinal;
}

double time_alltoallv_3step(int size, float* cpu_send_data, float* cpu_recv_data,
        float* gpu_data, cudaStream_t& stream, MPI_Comm& group_comm, int n_tests)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(group_comm, &num_procs);

    std::vector<int> send_sizes(num_procs, size);
    std::vector<int> recv_sizes(num_procs, size);
    std::vector<int> send_displs(num_procs+1);
    std::vector<int> recv_displs(num_procs+1);
    send_displs[0] = 0;
    recv_displs[0] = 0;
    for (int i = 0; i < num_procs; i++)
    {
        send_displs[i+1] = send_displs[i] + send_sizes[i];
        recv_displs[i+1] = recv_displs[i] + recv_sizes[i];
    }

    double t0, tfinal;
    int total_size = size * num_procs;
    int bytes = total_size * sizeof(float);
if (size == 1 && rank == 0) printf("Nmsgs %d, Bytes %d\n", num_procs, bytes);

    // Warm Up
    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        cudaMemcpyAsync(cpu_send_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        MPI_Alltoallv(cpu_send_data, send_sizes.data(), send_displs.data(), MPI_FLOAT,
                cpu_recv_data, recv_sizes.data(), recv_displs.data(), MPI_FLOAT, group_comm);
        cudaMemcpyAsync(gpu_data, cpu_recv_data, bytes, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;


    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        cudaMemcpyAsync(cpu_send_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        MPI_Alltoallv(cpu_send_data, send_sizes.data(), send_displs.data(), MPI_FLOAT,
                cpu_recv_data, recv_sizes.data(), recv_displs.data(), MPI_FLOAT, group_comm);
        cudaMemcpyAsync(gpu_data, cpu_recv_data, bytes, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;

    return tfinal;
}


double time_alltoallv_3step_imsg(int size, float* cpu_send_data, float* cpu_recv_data,
        float* gpu_data, cudaStream_t& stream, MPI_Comm& group_comm, int n_tests)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(group_comm, &num_procs);

    std::vector<int> send_procs(num_procs);
    std::vector<int> recv_procs(num_procs);
    for (int i = 0; i < num_procs; i++)
    {
        send_procs[i] = i;
        recv_procs[i] = i;
    }
    std::vector<MPI_Request> send_req(num_procs);
    std::vector<MPI_Request> recv_req(num_procs);

    double t0, tfinal;
    int total_size = size * num_procs;
    int bytes = total_size * sizeof(float);

    // Warm Up
    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        cudaMemcpyAsync(cpu_send_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        send_recv(size, num_procs, send_procs.data(), recv_procs.data(), 
            send_req.data(), recv_req.data(), cpu_send_data, cpu_recv_data,
            group_comm);
        cudaMemcpyAsync(gpu_data, cpu_recv_data, bytes, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;


    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        cudaMemcpyAsync(cpu_send_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        send_recv(size, num_procs, send_procs.data(), recv_procs.data(), 
            send_req.data(), recv_req.data(), cpu_send_data, cpu_recv_data,
            group_comm);
        cudaMemcpyAsync(gpu_data, cpu_recv_data, bytes, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;

    return tfinal;
}

double time_alltoallv_3step_msg(int size, float* cpu_send_data, float* cpu_recv_data,
       float* gpu_data, int ppg, int node_rank, cudaStream_t& stream, MPI_Comm& group_comm, 
       int n_tests)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(group_comm, &num_procs);

    double t0, tfinal;
    int gpu_rank = node_rank % ppg;
    int global_gpu = rank / ppg;
    bool master = gpu_rank == 0;
    int ping_tag = 1234;
    int pong_tag = 4321;
    MPI_Status status;

    int n_msgs = num_procs / ppg;
    int extra = num_procs % ppg;
    if (gpu_rank < extra) n_msgs++;

    int total_size = size * num_procs;
    int bytes = total_size * sizeof(float);

    std::vector<int> send_sizes(num_procs, 0);
    std::vector<int> send_displs(num_procs+1);
    std::vector<int> recv_sizes(num_procs, 0);
    std::vector<int> recv_displs(num_procs+1);
    send_displs[0] = 0;
    recv_displs[0] = 0;

    int proc = global_gpu + gpu_rank;
    if (proc >= num_procs) proc -= num_procs;
    for (int i = 0; i < n_msgs; i++)
    {
        send_sizes[proc] = size;
        proc += ppg;
        if (proc >= num_procs) proc -= num_procs;
    }
    proc = global_gpu - gpu_rank;
    if (proc < 0) proc += num_procs;
    for (int i = 0; i < n_msgs; i++)
    {
        recv_sizes[proc] = size;
        proc -= ppg;
        if (proc < 0) proc += num_procs;
    }
    for (int i = 0; i < num_procs; i++)
    {
        send_displs[i+1] = send_displs[i] + send_sizes[i];
        recv_displs[i+1] = recv_displs[i] + recv_sizes[i];
    }

    int nm, ctr, msg_size;
    
    // Warm Up 
    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        if (master)
        {
            cudaMemcpyAsync(cpu_send_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            ctr = n_msgs*size;
            for (int i = 1; i < ppg; i++)
            {
                nm = num_procs / ppg;
                if (extra > i) nm++;
                msg_size = nm * size;
                MPI_Send(&(cpu_send_data[ctr]), msg_size, MPI_FLOAT, rank+i, 
                        ping_tag, MPI_COMM_WORLD);
                ctr += msg_size;
            }
        }
        else
        {
            MPI_Recv(cpu_send_data, n_msgs*size, MPI_FLOAT, rank - gpu_rank, 
                    ping_tag, MPI_COMM_WORLD, &status);
        }

        MPI_Alltoallv(cpu_send_data, send_sizes.data(), send_displs.data(), MPI_FLOAT,
                cpu_recv_data, recv_sizes.data(), recv_displs.data(), MPI_FLOAT, group_comm);
      
        if (master)
        {
            ctr = n_msgs*size;
            for (int i = 1; i < ppg; i++)
            {
                nm = num_procs / ppg;
                if (extra > i) nm++;
                msg_size = nm * size;
                MPI_Recv(&(cpu_recv_data[ctr]), msg_size, MPI_FLOAT, rank+i, 
                       pong_tag, MPI_COMM_WORLD, &status);
                ctr += msg_size;
            }
            cudaMemcpyAsync(gpu_data, cpu_recv_data, bytes, cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
        }
        else
        {
           MPI_Send(cpu_recv_data, n_msgs*size, MPI_FLOAT, rank - gpu_rank, 
                  pong_tag, MPI_COMM_WORLD); 
        }
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;
    

    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        if (master)
        {
            cudaMemcpyAsync(cpu_send_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            ctr = n_msgs*size;
            for (int i = 1; i < ppg; i++)
            {
                nm = num_procs / ppg;
                if (extra > i) nm++;
                msg_size = nm * size;
                MPI_Send(&(cpu_send_data[ctr]), msg_size, MPI_FLOAT, rank+i, 
                        ping_tag, MPI_COMM_WORLD);
                ctr += msg_size;
            }
        }
        else
        {
            MPI_Recv(cpu_send_data, n_msgs*size, MPI_FLOAT, rank - gpu_rank, 
                    ping_tag, MPI_COMM_WORLD, &status);
        }

        MPI_Alltoallv(cpu_send_data, send_sizes.data(), send_displs.data(), MPI_FLOAT,
                cpu_recv_data, recv_sizes.data(), recv_displs.data(), MPI_FLOAT, group_comm);
      
        if (master)
        {
            ctr = n_msgs*size;
            for (int i = 1; i < ppg; i++)
            {
                nm = num_procs / ppg;
                if (extra > i) nm++;
                msg_size = nm * size;
                MPI_Recv(&(cpu_recv_data[ctr]), msg_size, MPI_FLOAT, rank+i, 
                       pong_tag, MPI_COMM_WORLD, &status);
                ctr += msg_size;
            }
            cudaMemcpyAsync(gpu_data, cpu_recv_data, bytes, cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
        }
        else
        {
           MPI_Send(cpu_recv_data, n_msgs*size, MPI_FLOAT, rank - gpu_rank, 
                  pong_tag, MPI_COMM_WORLD); 
        }
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;
    
    return tfinal;
}

double time_alltoallv_3step_msg_imsg(int size, float* cpu_send_data, float* cpu_recv_data,
       float* gpu_data, int ppg, int node_rank, cudaStream_t& stream, MPI_Comm& group_comm, 
       int n_tests)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(group_comm, &num_procs);

    double t0, tfinal;
    int gpu_rank = node_rank % ppg;
    int global_gpu = rank / ppg;
    bool master = gpu_rank == 0;
    int ping_tag = 1234;
    int pong_tag = 4321;
    MPI_Status status;

    int n_msgs = num_procs / ppg;
    int extra = num_procs % ppg;
    if (gpu_rank < extra) n_msgs++;

    int total_size = size * num_procs;
    int bytes = total_size * sizeof(float);

    std::vector<int> send_procs(n_msgs);
    std::vector<int> recv_procs(n_msgs);
    std::vector<MPI_Request> send_req(n_msgs);
    std::vector<MPI_Request> recv_req(n_msgs);

    int proc = global_gpu + gpu_rank;
    if (proc >= num_procs) proc -= num_procs;
    for (int i = 0; i < n_msgs; i++)
    {
        send_procs[i] = proc;
    }
    proc = global_gpu - gpu_rank;
    if (proc < 0) proc += num_procs;
    for (int i = 0; i < n_msgs; i++)
    {
        recv_procs[i] = proc;
    }

    int nm, ctr, msg_size;
    
    // Warm Up 
    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        if (master)
        {
            cudaMemcpyAsync(cpu_send_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            ctr = n_msgs*size;
            for (int i = 1; i < ppg; i++)
            {
                nm = num_procs / ppg;
                if (extra > i) nm++;
                msg_size = nm * size;
                MPI_Send(&(cpu_send_data[ctr]), msg_size, MPI_FLOAT, rank+i, 
                        ping_tag, MPI_COMM_WORLD);
                ctr += msg_size;
            }
        }
        else
        {
            MPI_Recv(cpu_send_data, n_msgs*size, MPI_FLOAT, rank - gpu_rank, 
                    ping_tag, MPI_COMM_WORLD, &status);
        }

        send_recv(size, n_msgs, send_procs.data(), recv_procs.data(), 
            send_req.data(), recv_req.data(), cpu_send_data, cpu_recv_data,
            group_comm);
      
        if (master)
        {
            ctr = n_msgs*size;
            for (int i = 1; i < ppg; i++)
            {
                nm = num_procs / ppg;
                if (extra > i) nm++;
                msg_size = nm * size;
                MPI_Recv(&(cpu_recv_data[ctr]), msg_size, MPI_FLOAT, rank+i, 
                       pong_tag, MPI_COMM_WORLD, &status);
                ctr += msg_size;
            }
            cudaMemcpyAsync(gpu_data, cpu_recv_data, bytes, cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
        }
        else
        {
           MPI_Send(cpu_recv_data, n_msgs*size, MPI_FLOAT, rank - gpu_rank, 
                  pong_tag, MPI_COMM_WORLD); 
        }
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;
    

    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        if (master)
        {
            cudaMemcpyAsync(cpu_send_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            ctr = n_msgs*size;
            for (int i = 1; i < ppg; i++)
            {
                nm = num_procs / ppg;
                if (extra > i) nm++;
                msg_size = nm * size;
                MPI_Send(&(cpu_send_data[ctr]), msg_size, MPI_FLOAT, rank+i, 
                        ping_tag, MPI_COMM_WORLD);
                ctr += msg_size;
            }
        }
        else
        {
            MPI_Recv(cpu_send_data, n_msgs*size, MPI_FLOAT, rank - gpu_rank, 
                    ping_tag, MPI_COMM_WORLD, &status);
        }

        send_recv(size, n_msgs, send_procs.data(), recv_procs.data(), 
            send_req.data(), recv_req.data(), cpu_send_data, cpu_recv_data,
            group_comm);
      
        if (master)
        {
            ctr = n_msgs*size;
            for (int i = 1; i < ppg; i++)
            {
                nm = num_procs / ppg;
                if (extra > i) nm++;
                msg_size = nm * size;
                MPI_Recv(&(cpu_recv_data[ctr]), msg_size, MPI_FLOAT, rank+i, 
                       pong_tag, MPI_COMM_WORLD, &status);
                ctr += msg_size;
            }
            cudaMemcpyAsync(gpu_data, cpu_recv_data, bytes, cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
        }
        else
        {
           MPI_Send(cpu_recv_data, n_msgs*size, MPI_FLOAT, rank - gpu_rank, 
                  pong_tag, MPI_COMM_WORLD); 
        }
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;
    
    return tfinal;
}


double time_alltoallv_3step_dup(int size, float* cpu_send_data, float* cpu_recv_data,
       float* gpu_data, int ppg, int node_rank, cudaStream_t& stream, MPI_Comm& group_comm, 
       int n_tests)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(group_comm, &num_procs);

    double t0, tfinal;
    int gpu_rank = node_rank % ppg;
    int global_gpu = rank / ppg;

    int n_msgs = num_procs / ppg;
    int extra = num_procs % ppg;
    if (gpu_rank < extra) n_msgs++;

    int total_size = size * n_msgs;
    int bytes = total_size * sizeof(float);

    std::vector<int> send_sizes(num_procs, 0);
    std::vector<int> send_displs(num_procs+1);
    std::vector<int> recv_sizes(num_procs, 0);
    std::vector<int> recv_displs(num_procs+1);
    send_displs[0] = 0;
    recv_displs[0] = 0;

    int proc = global_gpu + gpu_rank;
    if (proc >= num_procs) proc -= num_procs;
    for (int i = 0; i < n_msgs; i++)
    {
        send_sizes[proc] = size;
        proc += ppg;
        if (proc >= num_procs) proc -= num_procs;
    }
    proc = global_gpu - gpu_rank;
    if (proc < 0) proc += num_procs;
    for (int i = 0; i < n_msgs; i++)
    {
        recv_sizes[proc] = size;
        proc -= ppg;
        if (proc < 0) proc += num_procs;
    }
    for (int i = 0; i < num_procs; i++)
    {
        send_displs[i+1] = send_displs[i] + send_sizes[i];
        recv_displs[i+1] = recv_displs[i] + recv_sizes[i];
    }
    
    // Warm Up 
    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        cudaMemcpyAsync(cpu_send_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        MPI_Alltoallv(cpu_send_data, send_sizes.data(), send_displs.data(), MPI_FLOAT,
                cpu_recv_data, recv_sizes.data(), recv_displs.data(), MPI_FLOAT, group_comm);
        cudaMemcpyAsync(gpu_data, cpu_recv_data, bytes, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;
    

    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        cudaMemcpyAsync(cpu_send_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        MPI_Alltoallv(cpu_send_data, send_sizes.data(), send_displs.data(), MPI_FLOAT,
                cpu_recv_data, recv_sizes.data(), recv_displs.data(), MPI_FLOAT, group_comm);
        cudaMemcpyAsync(gpu_data, cpu_recv_data, bytes, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;

    return tfinal;
}


double time_alltoallv_3step_dup_imsg(int size, float* cpu_send_data, float* cpu_recv_data,
       float* gpu_data, int ppg, int node_rank, cudaStream_t& stream, MPI_Comm& group_comm, 
       int n_tests)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(group_comm, &num_procs);

    double t0, tfinal;
    int gpu_rank = node_rank % ppg;
    int global_gpu = rank / ppg;

    int n_msgs = num_procs / ppg;
    int extra = num_procs % ppg;
    if (gpu_rank < extra) n_msgs++;

    int total_size = size * n_msgs;
    int bytes = total_size * sizeof(float);

    std::vector<int> send_procs(n_msgs);
    std::vector<int> recv_procs(n_msgs);
    std::vector<MPI_Request> send_req(n_msgs);
    std::vector<MPI_Request> recv_req(n_msgs);
    
    int proc = global_gpu + gpu_rank;
    if (proc >= num_procs) proc -= num_procs;
    for (int i = 0; i < n_msgs; i++)
    {
        send_procs[i] = proc;
    }
    proc = global_gpu - gpu_rank;
    if (proc < 0) proc += num_procs;
    for (int i = 0; i < n_msgs; i++)
    {
        recv_procs[i] = proc;
    }
    
    // Warm Up 
    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        cudaMemcpyAsync(cpu_send_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        send_recv(size, n_msgs, send_procs.data(), recv_procs.data(), 
            send_req.data(), recv_req.data(), cpu_send_data, cpu_recv_data,
            group_comm);
        cudaMemcpyAsync(gpu_data, cpu_recv_data, bytes, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;
    

    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        cudaMemcpyAsync(cpu_send_data, gpu_data, bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        send_recv(size, n_msgs, send_procs.data(), recv_procs.data(), 
            send_req.data(), recv_req.data(), cpu_send_data, cpu_recv_data,
            group_comm);
        cudaMemcpyAsync(gpu_data, cpu_recv_data, bytes, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;

    return tfinal;
}

