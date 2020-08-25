cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(heterobench LANGUAGES CUDA CXX VERSION 0.1.0)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
set(heterobench_INCDIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

cmake_policy(SET CMP0104 OLD)

include(options)
include(testing)
include(source)

set(MPIRUN "mpirun" CACHE STRING "MPIRUN command")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "-ccbin=mpicxx -arch=sm_70")

set(GOOGLETEST_ROOT external/googletest CACHE STRING "Google Test source root")

include_directories(
    ${PROJECT_SOURCE_DIR}/${GOOGLETEST_ROOT}
    ${PROJECT_SOURCE_DIR}/${GOOGLETEST_ROOT}/include
    )

set(GOOGLETEST_SOURCES
    ${PROJECT_SOURCE_DIR}/${GOOGLETEST_ROOT}/src/gtest-all.cc
    ${PROJECT_SOURCE_DIR}/${GOOGLETEST_ROOT}/src/gtest_main.cc
    )


foreach(_source ${GOOGLETEST_SOURCES})
    set_source_files_properties(${_source} PROPERTIES GENERATED 1)
    #MESSAGE( STATUS "_source: "                 ${_source} )
endforeach()


add_library(googletest STATIC ${GOOGLETEST_SOURCES})
#///////////////////////////// end information of google test //////////////////////////////

add_subdirectory(src)


target_compile_options(heterobench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: # Flags for CUDA Objects
    -lineinfo
    -Xcompiler -Wall
    >
    $<$<COMPILE_LANGUAGE:CXX>: # Flags for C++ Objects
    -fno-omit-frame-pointer
    >
    )

set_property(TARGET heterobench PROPERTY CUDA_SEPARABLE_COMPILATION ON)

if (BUILD_EXAMPLES)
      add_subdirectory(examples)
endif()
