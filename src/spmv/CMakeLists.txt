# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called core_SOURCES containing all .cpp files:
set(spmv_HEADERS
    spmv/spmv.h
    spmv/spmv_timer.h
    spmv/spmv_profiler.h
    PARENT_SCOPE
    )

set(spmv_SOURCES
    spmv/spmv.cu
    spmv/spmv_timer.cu
    spmv/spmv_profiler.cu
    PARENT_SCOPE
    )
