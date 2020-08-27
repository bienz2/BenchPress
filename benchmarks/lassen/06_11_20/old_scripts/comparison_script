#!/bin/bash
#BSUB -J comparison_reuse
#BSUB -e comparison_reuse.%J.err
#BSUB -o comparison_reuse.%J.out
#BSUB -nnodes 2
#BSUB -q pdebug
#BSUB -W 00:15

#module load hwloc
#module load nsight-systems

cd /g/g14/bienz1/nodecomm

jsrun -a4 -c4 -g4 -r1 -n2 -M "-gpu" --latency_priority=gpu-cpu --print_placement=1 ./time_comparison




