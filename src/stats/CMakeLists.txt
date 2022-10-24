# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called core_SOURCES containing all .cpp files:
set(stats_HEADERS
    stats/stats.h
    PARENT_SCOPE
    )

if (USE_HIP)
    set(stats_SOURCES
        stats/stats_hip.cpp
        PARENT_SCOPE
        )
else()
    set(stats_SOURCES
        stats/stats.cu
        PARENT_SCOPE
        )
endif()
