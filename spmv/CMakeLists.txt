include_directories(${heterobench_INCDIR})

add_executable(mpi_sparse_mat mpi_sparse_mat.cu)
target_link_libraries(mpi_sparse_mat heterobench)

