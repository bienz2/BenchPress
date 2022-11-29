#!/bin/bash

export MPICH_GPU_SUPPORT_ENABLED=1

cd /g/g14/bienz1/BenchPress/tioga_build/examples

flux mini run --setop=mpibind=off -n 256 --verbose --exclusive --nodes=4 ./time_alltoallv
