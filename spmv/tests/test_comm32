#!/bin/bash
#BSUB -J comm_only_32
#BSUB -e comm_only_32.%J.err
#BSUB -o comm_only_32.%J.out
#BSUB -nnodes 32
#BSUB -W 00:30

module load gcc
module load cmake/3.18.1
module load cuda

cd /g/g14/bienz1/BenchPress/spmv

export NNODES=32
export NPROCS=8
export FILE=mpi_sparse_mat_comm_only
echo "8 Processes Per Node, 2 Per GPU:"

echo "GPU-Level Aggregation"
sh agg_gpu.sh

echo "Socket-Level Aggregation"
sh agg_socket.sh

echo "Node-Level Aggregation"
sh agg_node.sh


export NPROCS=12
echo "12 Processes Per Node, 3 Per GPU:"

echo "GPU-Level Aggregation"
sh agg_gpu.sh

echo "Socket-Level Aggregation"
sh agg_socket.sh

echo "Node-Level Aggregation"
sh agg_node.sh



export NPROCS=16
echo "16 Processes Per Node, 4 Per GPU:"

echo "GPU-Level Aggregation"
sh agg_gpu.sh

echo "Socket-Level Aggregation"
sh agg_socket.sh

echo "Node-Level Aggregation"
sh agg_node.sh




export NPROCS=20
echo "20 Processes Per Node, 5 Per GPU:"

echo "GPU-Level Aggregation"
sh agg_gpu.sh

echo "Socket-Level Aggregation"
sh agg_socket.sh

echo "Node-Level Aggregation"
sh agg_node.sh




export NPROCS=40
echo "40 Processes Per Node, 10 Per GPU:"

echo "GPU-Level Aggregation"
sh agg_gpu.sh

echo "Socket-Level Aggregation"
sh agg_socket.sh

echo "Node-Level Aggregation"
sh agg_node.sh
