all: mpi_sparse_mat

mpi_sparse_mat : mpi_sparse_mat.cu
	nvcc -ccbin=mpicxx -arch=sm_70 -I./raptor/raptor mpi_sparse_mat.cu -o mpi_sparse_mat -L./raptor/build/lib/ -lraptor -std=c++11 -lcusparse

clean : 
	rm mpi_sparse_mat
