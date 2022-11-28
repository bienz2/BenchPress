#!/bin/bash

export MPICH_GPU_SUPPORT_ENABLED=1

#hipcc -I/opt/cray/pe/mpich/8.1.20/ofi/gnu/9.1/include -o test test.cpp -L/opt/cray/pe/mpich/8.1.20/ofi/gnu/9.1/lib -lmpi -L/opt/cray/pe/mpich/8.1.20/gtl/lib/ -lmpi_gtl_hsa

mpicxx -D__HIP_PLATFORM_AMD__ -I/opt/rocm-5.3.0/include -I/opt/rocm-5.3.0/hip/include --rocm-path=/opt/rocm-5.3.0 -x hip -o test test.cpp -L/opt/cray/pe/mpich/8.1.20/gtl/lib -lmpi_gtl_hsa -Wl,-rpath,/opt/cray/pe/mpich/8.1.20/gtl/lib

#flux mini run --setop=mpibind=off -n 8 --verbose --exclusive --nodes=1 ./test
#flux mini run -N 1 -n 8 --exclusive --verbose ./test
