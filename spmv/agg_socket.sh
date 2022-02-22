#!/bin/bash

export PPS=$(expr ${NPROCS} / 2)
for level in {0..13}
do
   echo "Level ${level}"
    nvidia-cuda-mps-control -d
    jsrun -a${NPROCS} -c${NPROCS} -g4 -r1 -n${NNODES} -M "-gpu" --latency_priority=gpu-cpu --launch_distribution=packed ./${FILE} raptor/build/examples/diffusion/mat_${level}.pm ${PPS}
    echo quit | nvidia-cuda-mps-control
done

