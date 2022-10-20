cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(heterobench LANGUAGES CUDA CXX VERSION 0.1.0)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

set(PYTHONPATH "${PYTHONPATH}:${CMAKE_CURRENT_SOURCE_DIR}/plots/benchpress")
message("Add ${CMAKE_CURRENT_SOURCE_DIR}/plots/benchpress to PYTHONPATH in bash file")

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
set(heterobench_INCDIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

cmake_policy(SET CMP0104 OLD)

include(FeatureSummary)

option(SRUN "Running MPI Tests With srun" OFF)
option(APRUN "Running MPI Tests With aprun" OFF)
option(CUDA_AWARE "Test Performance of CUDA-Aware MPI" ON)

add_feature_info(srun SRUN "Run MPI tests with srun")
add_feature_info(aprun APRUN "Run MPI tests with aprun")
add_feature_info(cuda_aware CUDA_AWARE "Run CUDA-Aware tests")

include(options)
include(testing)
include(source)

set(MPIRUN "mpirun" CACHE STRING "MPIRUN command")

if (SRUN)
    set(MPIRUN "srun")
endif (SRUN)

if (APRUN)
    set(MPIRUN "aprun")
endif (APRUN)

if (CUDA_AWARE)
    add_definitions(-DCUDA_AWARE)
endif (CUDA_AWARE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "-ccbin=mpicxx -arch=sm_70")

##################### 
## GOOGLETEST      ##
#####################
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  # Specify the commit you depend on and update it regularly.
  GIT_TAG 5376968f6948923e2411081fd9372e71a59d8e77
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()
#####################



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
