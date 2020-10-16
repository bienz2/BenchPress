add_executable(test_allreduce test_allreduce.cpp)
target_link_libraries(test_allreduce heterobench ${MPI_LIBRARIES} googletest pthread )
add_test(AllreduceTest ${MPIRUN} -n 1 ./test_allreduce)
add_test(AllreduceTest ${MPIRUN} -n 4 ./test_allreduce)
add_test(AllreduceTest ${MPIRUN} -n 16 ./test_allreduce)

add_executable(test_alltoall test_alltoall.cpp)
target_link_libraries(test_alltoall heterobench ${MPI_LIBRARIES} googletest pthread )
add_test(AlltoallTest ${MPIRUN} -n 1 ./test_alltoall)
add_test(AlltoallTest ${MPIRUN} -n 4 ./test_alltoall)
add_test(AlltoallTest ${MPIRUN} -n 16 ./test_alltoall)


