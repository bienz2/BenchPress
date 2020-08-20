include_directories(${heterobench_INCDIR})

add_executable(time_memcpy time_memcpy.cpp)
target_link_libraries(time_memcpy heterobench)

add_executable(time_collective time_collective.cpp)
target_link_libraries(time_collective heterobench)

add_executable(time_ping_pong time_ping_pong.cpp)
target_link_libraries(time_ping_pong heterobench)

