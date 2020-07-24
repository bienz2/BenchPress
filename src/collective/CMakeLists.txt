# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called core_SOURCES containing all .cpp files:
set(collective_HEADERS
    collective/collective_timer.h
    collective/collective_profiler.h
    PARENT_SCOPE
    )

set(collective_SOURCES
    collective/collective_timer.cu
    collective/collective_profiler.cu
    PARENT_SCOPE
    )
