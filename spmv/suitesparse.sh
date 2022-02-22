#!/bin/bash


export MAT=Flan_1565.pm
echo ${MAT}

echo "No Aggregation"
export PPG=$(expr ${NPROCS} / 4)
nvidia-cuda-mps-control -d
jsrun -a${NPROCS} -c${NPROCS} -g4 -r1 -n${NNODES} -M "-gpu" --latency_priority=gpu-cpu --launch_distribution=packed ./${FILE} /p/gpfs1/bienz1/${MAT} ${PPG}
echo quit | nvidia-cuda-mps-control


echo "Socket Aggregation"
export PPS=$(expr ${NPROCS} / 2)
nvidia-cuda-mps-control -d
jsrun -a${NPROCS} -c${NPROCS} -g4 -r1 -n${NNODES} -M "-gpu" --latency_priority=gpu-cpu --launch_distribution=packed ./${FILE} /p/gpfs1/bienz1/${MAT} ${PPS}
echo quit | nvidia-cuda-mps-control


echo "Node Aggregation"
nvidia-cuda-mps-control -d
jsrun -a${NPROCS} -c${NPROCS} -g4 -r1 -n${NNODES} -M "-gpu" --latency_priority=gpu-cpu --launch_distribution=packed ./${FILE} /p/gpfs1/bienz1/${MAT} ${NPROCS}
echo quit | nvidia-cuda-mps-control


