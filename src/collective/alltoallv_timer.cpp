#include "alltoallv_timer.hpp"


/*******************************************************************
 *** Method : send_recv(...)
 ***
 ***    size : int
 ***        The size of each MPI\_Alltoallv method
 ***    n_msgs : int
 ***        The number of messages (number of processes active in Alltoallv)
 ***    send_procs : int*
 ***        The processes to which this rank will send
 ***    recv_procs : int*
 ***        The processes from which this rank will recv
 ***    send_req : MPI_Request*
 ***        MPI_Requests for each of the sends
 ***    recv_req : MPI_Request*
 ***        MPI_Requests for each of the recvs
 ***    send_data : float*
 ***        Data to be sent
 ***    recv_data : float*
 ***        Array in which data will be received
 ***    comm : MPI_Comm
 ***        MPI_Communicator for alltoallv
 ***    tag : int (default 83205)
 ***        Tag of messages
 ***
 ***    This method performs an MPI\_Alltoallv with the information 
 ***    passed in the arguments.  A chunk of 'size' floats is sent from
 ***    'send_data' to each of the processes in 'send_procs', and a chunk
 ***    of 'size' floats is received into 'recv_data' from each of the
 ***    processes in 'recv_procs'
*******************************************************************/ 
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

/*******************************************************************
 *** Method : time_alltoallv(...)
 ***
 ***    size : int
 ***        The size of each MPI\_Alltoallv method
 ***    gpu_send_data : float*
 ***        Data to be sent in alltoallv, in GPU memory
 ***    gpu_recv_data : float*
 ***        Array in which data is to be received, in GPU memory
 ***    group_comm : MPI_Comm
 ***        MPI_Communicator on which alltoallv is performed
 ***    n_tests : int
 ***        The number of iterations of alltoallv (for timer precision)
 ***
 ***    This method times the cost of a single alltoallv operation, 
 ***    using a CUDA-Aware call to MPI\_Alltoallv(...).  An MPI\_Alltoallv
 ***    operation is called on 'gpu_send_data', and data is received
 ***    into 'gpu_recv_data'
*******************************************************************/ 
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
    gpuDeviceSynchronize();
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        MPI_Alltoallv(gpu_send_data, send_sizes.data(), send_displs.data(), MPI_FLOAT,
                gpu_recv_data, recv_sizes.data(), recv_displs.data(), MPI_FLOAT, group_comm);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;


    gpuDeviceSynchronize();
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


/*******************************************************************
 *** Method : time_alltoallv_imsg(...)
 ***
 ***    size : int
 ***        The size of each send_recv method
 ***    gpu_send_data : float*
 ***        Data to be sent with MPI\_Isends, in GPU memory
 ***    gpu_recv_data : float*
 ***        Array in which data is to be received, in GPU memory
 ***    group_comm : MPI_Comm
 ***        MPI_Communicator on which send_recv is performed
 ***    n_tests : int
 ***        The number of iterations of send_recv (for timer precision)
 ***
 ***    This method times the cost of a single send_recv(...) operation, 
 ***    using a CUDA-Aware call to each MPI\_Isend, MPI\_Irecv, and MPI\_Waitall.
 ***    The send_recv() operation is called on 'gpu_send_data, and data is 
 ***    received into 'gpu_recv_data'
*******************************************************************/ 
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
    gpuDeviceSynchronize();
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        send_recv(size, num_procs, send_procs.data(), recv_procs.data(),
                send_req.data(), recv_req.data(), gpu_send_data, gpu_recv_data,
                group_comm);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;


    gpuDeviceSynchronize();
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


/*******************************************************************
 *** Method : time_alltoallv_3step(...)
 ***
 ***    size : int
 ***        The size of each MPI\_Alltoallv
 ***    cpu_send_data : float*
 ***        Data to be sent with MPI\_Alltoallv, in CPU memory
 ***    cpu_recv_data : float*
 ***        Array in which data is to be received, in CPU memory
 ***    gpu_data : float*
 ***        Data will original data on GPU, where received data will
 ***        be placed. (TODO : Currently overwrites GPU data)
 ***    stream : gpuStream_t&
 ***        Cuda Stream on which data is copied.  
 ***    group_comm : MPI_Comm
 ***        MPI_Communicator on which MPI\_Alltoallv is performed
 ***    n_tests : int
 ***        The number of iterations of MPI\_Alltoallv (for timer precision)
 ***
 ***    This method times the cost of copying data to a single CPU and 
 ***    performing an MPI\_Alltoallv operation on this data, in CPU memory.
*******************************************************************/ 
double time_alltoallv_3step(int size, float* cpu_send_data, float* cpu_recv_data,
        float* gpu_data, gpuStream_t& stream, MPI_Comm& group_comm, int n_tests)
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
    gpuDeviceSynchronize();
    gpuStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        gpuMemcpyAsync(cpu_send_data, gpu_data, bytes, gpuMemcpyDeviceToHost, stream);
        gpuStreamSynchronize(stream);
        MPI_Alltoallv(cpu_send_data, send_sizes.data(), send_displs.data(), MPI_FLOAT,
                cpu_recv_data, recv_sizes.data(), recv_displs.data(), MPI_FLOAT, group_comm);
        gpuMemcpyAsync(gpu_data, cpu_recv_data, bytes, gpuMemcpyHostToDevice, stream);
        gpuStreamSynchronize(stream);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;


    gpuDeviceSynchronize();
    gpuStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        gpuMemcpyAsync(cpu_send_data, gpu_data, bytes, gpuMemcpyDeviceToHost, stream);
        gpuStreamSynchronize(stream);
        MPI_Alltoallv(cpu_send_data, send_sizes.data(), send_displs.data(), MPI_FLOAT,
                cpu_recv_data, recv_sizes.data(), recv_displs.data(), MPI_FLOAT, group_comm);
        gpuMemcpyAsync(gpu_data, cpu_recv_data, bytes, gpuMemcpyHostToDevice, stream);
        gpuStreamSynchronize(stream);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;

    return tfinal;
}


/*******************************************************************
 *** Method : time_alltoallv_3step_imsg(...)
 ***
 ***    size : int
 ***        The size of each send_recv method
 ***    cpu_send_data : float*
 ***        Data to be sent with each MPI\_Isend, in CPU memory
 ***    cpu_recv_data : float*
 ***        Array in which data is to be received, in CPU memory
 ***    gpu_data : float*
 ***        Data will original data on GPU, where received data will
 ***        be placed. (TODO : Currently overwrites GPU data)
 ***    stream : gpuStream_t&
 ***        Cuda Stream on which data is copied.  
 ***    group_comm : MPI_Comm
 ***        MPI_Communicator on which send_recv is performed
 ***    n_tests : int
 ***        The number of iterations of send_recv (for timer precision)
 ***
 ***    This method times the cost of copying data to a single CPU and 
 ***    performing a send_recv operation on this data, in CPU memory.
*******************************************************************/ 
double time_alltoallv_3step_imsg(int size, float* cpu_send_data, float* cpu_recv_data,
        float* gpu_data, gpuStream_t& stream, MPI_Comm& group_comm, int n_tests)
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
    gpuDeviceSynchronize();
    gpuStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        gpuMemcpyAsync(cpu_send_data, gpu_data, bytes, gpuMemcpyDeviceToHost, stream);
        gpuStreamSynchronize(stream);
        send_recv(size, num_procs, send_procs.data(), recv_procs.data(), 
            send_req.data(), recv_req.data(), cpu_send_data, cpu_recv_data,
            group_comm);
        gpuMemcpyAsync(gpu_data, cpu_recv_data, bytes, gpuMemcpyHostToDevice, stream);
        gpuStreamSynchronize(stream);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;


    gpuDeviceSynchronize();
    gpuStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        gpuMemcpyAsync(cpu_send_data, gpu_data, bytes, gpuMemcpyDeviceToHost, stream);
        gpuStreamSynchronize(stream);
        send_recv(size, num_procs, send_procs.data(), recv_procs.data(), 
            send_req.data(), recv_req.data(), cpu_send_data, cpu_recv_data,
            group_comm);
        gpuMemcpyAsync(gpu_data, cpu_recv_data, bytes, gpuMemcpyHostToDevice, stream);
        gpuStreamSynchronize(stream);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;

    return tfinal;
}



/*******************************************************************
 *** Method : time_alltoallv_3step_msg(...)
 ***
 ***    size : int
 ***        The size of each MPI\_Alltoallv
 ***    cpu_send_data : float*
 ***        Data to be sent with MPI\_Alltoallv, in CPU memory
 ***    cpu_recv_data : float*
 ***        Array in which data is to be received, in CPU memory
 ***    gpu_data : float*
 ***        Data will original data on GPU, where received data will
 ***        be placed. (TODO : Currently overwrites GPU data)
 ***    ppg : int
 ***        Number of processes per GPU
 ***    node_rank : int
 ***        Local rank on node 
 ***        (e.g. if rank is 50, PPN is 40, then node_rank is 10)
 ***    stream : gpuStream_t&
 ***        Cuda Stream on which data is copied.  
 ***    group_comm : MPI_Comm
 ***        MPI_Communicator on which MPI\_Alltoallv is performed
 ***    n_tests : int
 ***        The number of iterations of MPI\_Alltoallv (for timer precision)
 ***
 ***    This method times the cost of copying data to a single CPU,
 ***    redistributing the data to all available CPU cores, and 
 ***    performing an MPI\_Alltoallv operation on this data, in CPU memory.
*******************************************************************/ 
double time_alltoallv_3step_msg(int size, float* cpu_send_data, float* cpu_recv_data,
       float* gpu_data, int ppg, int node_rank, gpuStream_t& stream, MPI_Comm& group_comm, 
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
    gpuDeviceSynchronize();
    gpuStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        if (master)
        {
            gpuMemcpyAsync(cpu_send_data, gpu_data, bytes, gpuMemcpyDeviceToHost, stream);
            gpuStreamSynchronize(stream);

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
            gpuMemcpyAsync(gpu_data, cpu_recv_data, bytes, gpuMemcpyHostToDevice, stream);
            gpuStreamSynchronize(stream);
        }
        else
        {
           MPI_Send(cpu_recv_data, n_msgs*size, MPI_FLOAT, rank - gpu_rank, 
                  pong_tag, MPI_COMM_WORLD); 
        }
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;
    

    gpuDeviceSynchronize();
    gpuStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        if (master)
        {
            gpuMemcpyAsync(cpu_send_data, gpu_data, bytes, gpuMemcpyDeviceToHost, stream);
            gpuStreamSynchronize(stream);

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
            gpuMemcpyAsync(gpu_data, cpu_recv_data, bytes, gpuMemcpyHostToDevice, stream);
            gpuStreamSynchronize(stream);
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


/*******************************************************************
 *** Method : time_alltoallv_3step_msg_imsg(...)
 ***
 ***    size : int
 ***        The size of each send_recv operation
 ***    cpu_send_data : float*
 ***        Data to be sent with each MPI\_Isend, in CPU memory
 ***    cpu_recv_data : float*
 ***        Array in which data is to be received, in CPU memory
 ***    gpu_data : float*
 ***        Data will original data on GPU, where received data will
 ***        be placed. (TODO : Currently overwrites GPU data)
 ***    ppg : int
 ***        Number of processes per GPU
 ***    node_rank : int
 ***        Local rank on node 
 ***        (e.g. if rank is 50, PPN is 40, then node_rank is 10)
 ***    stream : gpuStream_t&
 ***        Cuda Stream on which data is copied.  
 ***    group_comm : MPI_Comm
 ***        MPI_Communicator on which send_recv is performed
 ***    n_tests : int
 ***        The number of iterations of send_recv (for timer precision)
 ***
 ***    This method times the cost of copying data to a single CPU,
 ***    redistributing the data to all available CPU cores, and 
 ***    performing a send_recv operation on this data, in CPU memory.
*******************************************************************/ 
double time_alltoallv_3step_msg_imsg(int size, float* cpu_send_data, float* cpu_recv_data,
       float* gpu_data, int ppg, int node_rank, gpuStream_t& stream, MPI_Comm& group_comm, 
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
    gpuDeviceSynchronize();
    gpuStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        if (master)
        {
            gpuMemcpyAsync(cpu_send_data, gpu_data, bytes, gpuMemcpyDeviceToHost, stream);
            gpuStreamSynchronize(stream);

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
            gpuMemcpyAsync(gpu_data, cpu_recv_data, bytes, gpuMemcpyHostToDevice, stream);
            gpuStreamSynchronize(stream);
        }
        else
        {
           MPI_Send(cpu_recv_data, n_msgs*size, MPI_FLOAT, rank - gpu_rank, 
                  pong_tag, MPI_COMM_WORLD); 
        }
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;
    

    gpuDeviceSynchronize();
    gpuStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        if (master)
        {
            gpuMemcpyAsync(cpu_send_data, gpu_data, bytes, gpuMemcpyDeviceToHost, stream);
            gpuStreamSynchronize(stream);

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
            gpuMemcpyAsync(gpu_data, cpu_recv_data, bytes, gpuMemcpyHostToDevice, stream);
            gpuStreamSynchronize(stream);
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


/*******************************************************************
 *** Method : time_alltoallv_3step_dup(...)
 ***
 ***    size : int
 ***        The size of each MPI\_Alltoallv
 ***    cpu_send_data : float*
 ***        Data to be sent with MPI\_Alltoallv, in CPU memory
 ***    cpu_recv_data : float*
 ***        Array in which data is to be received, in CPU memory
 ***    gpu_data : float*
 ***        Data will original data on GPU, where received data will
 ***        be placed. (TODO : Currently overwrites GPU data)
 ***    ppg : int
 ***        Number of processes per GPU
 ***    node_rank : int
 ***        Local rank on node 
 ***        (e.g. if rank is 50, PPN is 40, then node_rank is 10)
 ***    stream : gpuStream_t&
 ***        Cuda Stream on which data is copied.  
 ***    group_comm : MPI_Comm
 ***        MPI_Communicator on which MPI\_Alltoallv is performed
 ***    n_tests : int
 ***        The number of iterations of MPI\_Alltoallv (for timer precision)
 ***
 ***    This method times the cost of copying a portion of the data 
 ***    directly to each available CPU core and performing an MPI\_Alltoallv
 ***    operation on this data, in CPU memory.  This does not include the cost
 ***    of sending the device pointer to other processes, as that has a one 
 ***    time cost at setup.   Each CPU core calls gpuMemcpyAsync on its
 ***    individual gpuStream.
*******************************************************************/ 
double time_alltoallv_3step_dup(int size, float* cpu_send_data, float* cpu_recv_data,
       float* gpu_data, int ppg, int node_rank, gpuStream_t& stream, MPI_Comm& group_comm, 
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
    gpuDeviceSynchronize();
    gpuStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        gpuMemcpyAsync(cpu_send_data, gpu_data, bytes, gpuMemcpyDeviceToHost, stream);
        gpuStreamSynchronize(stream);
        MPI_Alltoallv(cpu_send_data, send_sizes.data(), send_displs.data(), MPI_FLOAT,
                cpu_recv_data, recv_sizes.data(), recv_displs.data(), MPI_FLOAT, group_comm);
        gpuMemcpyAsync(gpu_data, cpu_recv_data, bytes, gpuMemcpyHostToDevice, stream);
        gpuStreamSynchronize(stream);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;
    

    gpuDeviceSynchronize();
    gpuStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        gpuMemcpyAsync(cpu_send_data, gpu_data, bytes, gpuMemcpyDeviceToHost, stream);
        gpuStreamSynchronize(stream);
        MPI_Alltoallv(cpu_send_data, send_sizes.data(), send_displs.data(), MPI_FLOAT,
                cpu_recv_data, recv_sizes.data(), recv_displs.data(), MPI_FLOAT, group_comm);
        gpuMemcpyAsync(gpu_data, cpu_recv_data, bytes, gpuMemcpyHostToDevice, stream);
        gpuStreamSynchronize(stream);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;

    return tfinal;
}



/*******************************************************************
 *** Method : time_alltoallv_3step_dup_imsg(...)
 ***
 ***    size : int
 ***        The size of each send_recv operation
 ***    cpu_send_data : float*
 ***        Data to be sent with each  MPI\_Isend, in CPU memory
 ***    cpu_recv_data : float*
 ***        Array in which data is to be received, in CPU memory
 ***    gpu_data : float*
 ***        Data will original data on GPU, where received data will
 ***        be placed. (TODO : Currently overwrites GPU data)
 ***    ppg : int
 ***        Number of processes per GPU
 ***    node_rank : int
 ***        Local rank on node 
 ***        (e.g. if rank is 50, PPN is 40, then node_rank is 10)
 ***    stream : gpuStream_t&
 ***        Cuda Stream on which data is copied.  
 ***    group_comm : MPI_Comm
 ***        MPI_Communicator on which send_recv is performed
 ***    n_tests : int
 ***        The number of iterations of send_recv (for timer precision)
 ***
 ***    This method times the cost of copying a portion of the data 
 ***    directly to each available CPU core and performing a send_recv
 ***    operation on this data, in CPU memory.  This does not include the cost
 ***    of sending the device pointer to other processes, as that has a one 
 ***    time cost at setup.   Each CPU core calls gpuMemcpyAsync on its
 ***    individual gpuStream.
*******************************************************************/ 
double time_alltoallv_3step_dup_imsg(int size, float* cpu_send_data, float* cpu_recv_data,
       float* gpu_data, int ppg, int node_rank, gpuStream_t& stream, MPI_Comm& group_comm, 
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
    gpuDeviceSynchronize();
    gpuStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        gpuMemcpyAsync(cpu_send_data, gpu_data, bytes, gpuMemcpyDeviceToHost, stream);
        gpuStreamSynchronize(stream);
        send_recv(size, n_msgs, send_procs.data(), recv_procs.data(), 
            send_req.data(), recv_req.data(), cpu_send_data, cpu_recv_data,
            group_comm);
        gpuMemcpyAsync(gpu_data, cpu_recv_data, bytes, gpuMemcpyHostToDevice, stream);
        gpuStreamSynchronize(stream);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;
    

    gpuDeviceSynchronize();
    gpuStreamSynchronize(stream);
    MPI_Barrier(group_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        gpuMemcpyAsync(cpu_send_data, gpu_data, bytes, gpuMemcpyDeviceToHost, stream);
        gpuStreamSynchronize(stream);
        send_recv(size, n_msgs, send_procs.data(), recv_procs.data(), 
            send_req.data(), recv_req.data(), cpu_send_data, cpu_recv_data,
            group_comm);
        gpuMemcpyAsync(gpu_data, cpu_recv_data, bytes, gpuMemcpyHostToDevice, stream);
        gpuStreamSynchronize(stream);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;

    return tfinal;
}

