# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called core_SOURCES containing all .cpp files:
set(stats_HEADERS
    stats/stats.hpp
    PARENT_SCOPE
    )

set(stats_SOURCES
    stats/stats.cpp
    PARENT_SCOPE
    )
