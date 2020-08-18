add_executable(test_collective test_collective.cpp)
target_link_libraries(test_collective heterobench ${MPI_LIBRARIES} googletest pthread )
add_test(CollectiveTest ${MPIRUN} -n 1 ./test_collective)
add_test(CollectiveTest ${MPIRUN} -n 4 ./test_collective)
add_test(CollectiveTest ${MPIRUN} -n 16 ./test_collective)


