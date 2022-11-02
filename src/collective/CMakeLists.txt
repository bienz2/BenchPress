# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called core_SOURCES containing all .cpp files:
set(collective_HEADERS
    collective/allreduce_timer.hpp
    collective/alltoall_timer.hpp
    collective/alltoallv_timer.hpp
    collective/allreduce_profiler.hpp
    collective/alltoall_profiler.hpp
    collective/alltoallv_profiler.hpp
    PARENT_SCOPE
    )

set(collective_SOURCES)
    collective/allreduce_timer.cpp
    collective/alltoall_timer.cpp
    collective/alltoallv_timer.cpp
    collective/allreduce_profiler.cpp
    collective/alltoall_profiler.cpp
    collective/alltoallv_profiler.cpp
    PARENT_SCOPE
    )
        
