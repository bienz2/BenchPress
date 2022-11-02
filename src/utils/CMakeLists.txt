# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called core_SOURCES containing all .cpp files:
if (USE_HIP)
    set(utils_HEADERS
	utils/utils.hpp
        utils/utils_hip.hpp
        PARENT_SCOPE
        )
else()
    set(utils_HEADERS
	utils/utils.hpp
        utils/utils_cuda.hpp
        PARENT_SCOPE
        )
endif()
