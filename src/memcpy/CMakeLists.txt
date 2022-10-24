# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called core_SOURCES containing all .cpp files:
set(memcpy_HEADERS
    memcpy/memcpy_timer.h
    memcpy/memcpy_profiler.h
    PARENT_SCOPE
    )

if (USE_HIP)
    set(memcpy_SOURCES
        memcpy/memcpy_timer_hip.cpp
        memcpy/memcpy_profiler_hip.cpp
        PARENT_SCOPE
        )
else()
    set(memcpy_SOURCES
        memcpy/memcpy_timer.cu
        memcpy/memcpy_profiler.cu
        PARENT_SCOPE
        )
endif()
