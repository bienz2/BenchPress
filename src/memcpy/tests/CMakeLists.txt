add_executable(test_memcpy test_memcpy.cpp)
target_link_libraries(test_memcpy heterobench ${MPI_LIBRARIES} googletest pthread )
add_test(MemcpyTest ${MPIRUN} -n 1 ./test_memcpy)
add_test(MemcpyTest ${MPIRUN} -n 4 ./test_memcpy)
add_test(MemcpyTest ${MPIRUN} -n 16 ./test_memcpy)


