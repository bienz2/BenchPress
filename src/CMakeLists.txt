include_directories(.)

if (NOT USE_HIP)
    find_package(CUDAToolkit REQUIRED)
endif(NOT USE_HIP)

add_subdirectory(utils)
add_subdirectory(collective)
add_subdirectory(memcpy)
add_subdirectory(ping_pong)
add_subdirectory(stats)

if(NOT USE_HIP)
    set_source_files_properties(${collective_SOURCES} ${memcpy_SOURCES} ${ping_pong_SOURCES} ${spmv_SOURCES} ${stats_SOURCES} PROPERTIES LANGUAGE CUDA)
endif()

add_library(benchpress benchpress.hpp ${utils_HEADERS}
    ${collective_SOURCES} ${collective_HEADERS}
    ${memcpy_SOURCES} ${memcpy_HEADERS}
    ${ping_pong_SOURCES} ${ping_pong_HEADERS}
    ${spmv_SOURCES} ${spmv_HEADERS} 
    ${stats_SOURCES} ${stats_HEADERS}
)

target_link_libraries(benchpress ${MPI_LIBRARIES} ${EXTERNAL_LIBS})

if (NOT USE_HIP)
    target_link_libraries(benchpress CUDA::cudart)
endif(NOT USE_HIP)


install(TARGETS benchpress DESTINATION "lib")
install(FILES benchpress.hpp DESTINATION "include/BenchPress")
install(FILES ${utils_HEADERS} DESTINATION "include/BenchPress/utils")
install(FILES ${collective_HEADERS} DESTINATION "include/BenchPress/collective") 
install(FILES ${memcpy_HEADERS} DESTINATION "include/BenchPress/memcpy")
install(FILES ${ping_pong_HEADERS} DESTINATION "include/BenchPress/ping_pong")
install(FILES ${spmv_HEADERS} DESTINATION "include/BenchPress/spmv")
install(FILES ${stats_HEADERS} DESTINATION "include/BenchPress/stats")


include_directories(".")
if(ENABLE_UNIT_TESTS)
    add_subdirectory(memcpy/tests)
    add_subdirectory(ping_pong/tests)
    add_subdirectory(collective/tests)
endif()
