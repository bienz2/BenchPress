include_directories(.)

add_subdirectory(collective)
add_subdirectory(memcpy)
add_subdirectory(ping_pong)

add_library(heterobench STATIC 
    ${collective_SOURCES} ${collective_HEADERS}
    ${memcpy_SOURCES} ${memcpy_HEADERS}
    ${ping_pong_SOURCES} ${ping_pong_HEADERS})

target_link_libraries(heterobench ${EXTERNAL_LIBS})

install(TARGETS heterobench DESTINATION "lib")
install(FILES heterobench.hpp DESTINATION "include/HeteroBench")
install(FILES ${collective_HEADERS} DESTINATION "include/HeteroBench/collective") 
install(FILES ${memcpy_HEADERS} DESTINATION "include/HeteroBench/memcpy")
install(FILES ${ping_pong_HEADERS} DESTINATION "include/HeteroBench/ping_pong")


include_directories(".")
if(ENABLE_UNIT_TESTS)
    add_subdirectory(memcpy/tests)
    add_subdirectory(ping_pong/tests)
endif()
