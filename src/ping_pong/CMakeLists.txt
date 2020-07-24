# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called core_SOURCES containing all .cpp files:
set(ping_pong_HEADERS
    ping_pong/ping_pong_timer.h
    ping_pong/ping_pong_profiler.h
    PARENT_SCOPE
    )

set(ping_pong_SOURCES
    ping_pong/ping_pong_timer.cu
    ping_pong/ping_pong_profiler.cu
    PARENT_SCOPE
    )
