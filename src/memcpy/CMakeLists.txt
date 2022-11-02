# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called core_SOURCES containing all .cpp files:
set(memcpy_HEADERS
    memcpy/memcpy_timer.hpp
    memcpy/memcpy_profiler.hpp
    PARENT_SCOPE
    )

set(memcpy_SOURCES
    memcpy/memcpy_timer.cpp
    memcpy/memcpy_profiler.cpp
    PARENT_SCOPE
    )
