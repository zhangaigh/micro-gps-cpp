message( STATUS "Building for:  " ${CMAKE_SYSTEM} "(" ${CMAKE_SYSTEM_PROCESSOR} ")" )
option(ON_VISIONGPU1 "Option description" ON)
add_definitions(-DON_VISIONGPU1)

set(CMAKE_CXX_FLAGS "-std=c++11")

set(EXECUTABLE_OUTPUT_PATH ${MY_DIR}/bin)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})
link_directories(${OpenCV_LIBS_DIR})


# Eigen
set(Eigen3_INCLUDE_DIR /usr/include/eigen3)
# find_package(Eigen3 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIR})

# FLANN
# set(Flann_INCLUDE_DIR /usr/local/include/flann)
# set(Flann_LIBS_DIR /usr/local/lib)
set(Flann_INCLUDE_DIR /Users/lgzhang/Dropbox/Research/micro_gps/code/flann-1.8.4-src/install/include)
set(Flann_LIBS_DIR /Users/lgzhang/Dropbox/Research/micro_gps/code/flann-1.8.4-src/install/lib)
include_directories(${Flann_INCLUDE_DIR})
link_directories(${Flann_LIBS_DIR})

# GLFW
# find_package(GLFW REQUIRED)
set(${GLFW_INCLUDE_DIR} /usr/local/include)
set(${GLFW_LIBS_DIR} /usr/local/lib)
include_directories(${GLFW_INCLUDE_DIR})
link_directories(${GLFW_LIBS_DIR})

# # VLFEAT
# set(VLFEAT_INCLUDE_DIR ${MY_DIR}/vlfeat)
# set(VLFEAT_LIBS_DIR ${MY_DIR}/vlfeat/bin/glnxa64)
# include_directories(${VLFEAT_INCLUDE_DIR})
# link_directories(${VLFEAT_LIBS_DIR})

# SiftGPU
set(SiftGPU_INCLUDE_DIR /home/linguangzhang/Documents/micro-gps/SiftGPU/src/SiftGPU)
set(SiftGPU_LIBS_DIR /home/linguangzhang/Documents/micro-gps/SiftGPU/bin)
include_directories(${SiftGPU_INCLUDE_DIR})
link_directories(${SiftGPU_LIBS_DIR})

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib)


# GFLAGS
# find_package(gflags REQUIRED)
# message(STATUS "gflags_include = " ${gflags_INCLUDE_DIR})
# message(STATUS ${gflags_LIBS_DIR})
include_directories(/home/linguangzhang/Documents/micro-gps/gflags/build/include)
 link_directories(/home/linguangzhang/Documents/micro-gps/gflags/build/lib)
