include_directories(.)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(collective)
add_subdirectory(memcpy)
add_subdirectory(ping_pong)
#add_subdirectory(spmv)

add_library(heterobench heterobench.hpp 
    ${collective_SOURCES} ${collective_HEADERS}
    ${memcpy_SOURCES} ${memcpy_HEADERS}
    ${ping_pong_SOURCES} ${ping_pong_HEADERS}
    ${spmv_SOURCES} ${spmv_HEADERS} 
)

target_link_libraries(heterobench ${MPI_LIBRARIES} CUDA::cudart ${EXTERNAL_LIBS})
#set_property(TARGET heterobench PROPERTY CUDA_ARCHITECTURES 70)

install(TARGETS heterobench DESTINATION "lib")
install(FILES heterobench.hpp DESTINATION "include/BenchPress")
install(FILES ${collective_HEADERS} DESTINATION "include/BenchPress/collective") 
install(FILES ${memcpy_HEADERS} DESTINATION "include/BenchPress/memcpy")
install(FILES ${ping_pong_HEADERS} DESTINATION "include/BenchPress/ping_pong")
install(FILES ${spmv_HEADERS} DESTINATION "include/BenchPress/spmv")


include_directories(".")
if(ENABLE_UNIT_TESTS)
    add_subdirectory(memcpy/tests)
    add_subdirectory(ping_pong/tests)
    add_subdirectory(collective/tests)
    #add_subdirectory(spmv/tests)
endif()
