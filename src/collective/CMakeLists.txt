# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called core_SOURCES containing all .cpp files:
set(collective_HEADERS
    collective/allreduce_timer.h
    collective/alltoall_timer.h
    collective/alltoallv_timer.h
    collective/allreduce_profiler.h
    collective/alltoall_profiler.h
    collective/alltoallv_profiler.h
    PARENT_SCOPE
    )

set(collective_SOURCES
    collective/allreduce_timer.cu
    collective/alltoall_timer.cu
    collective/alltoallv_timer.cu
    collective/allreduce_profiler.cu
    collective/alltoall_profiler.cu
    collective/alltoallv_profiler.cu
    PARENT_SCOPE
    )
