# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called core_SOURCES containing all .cpp files:
set(ping_pong_HEADERS
    ping_pong/ping_pong_timer.hpp
    ping_pong/ping_pong_profiler.hpp
    PARENT_SCOPE
    )

set(ping_pong_SOURCES
    ping_pong/ping_pong_timer.cpp
    ping_pong/ping_pong_profiler.cpp
    PARENT_SCOPE
    )
