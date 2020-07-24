add_executable(test_ping_pong test_ping_pong.cpp)
target_link_libraries(test_ping_pong heterobench ${MPI_LIBRARIES} googletest pthread )
add_test(PingPongTest ${MPIRUN} -n 1 ./test_ping_pong)
add_test(PingPongTest ${MPIRUN} -n 4 ./test_ping_pong)
add_test(PingPongTest ${MPIRUN} -n 16 ./test_ping_pong)


