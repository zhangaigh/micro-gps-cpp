message( STATUS "Building for:  " ${CMAKE_SYSTEM} "(" ${CMAKE_SYSTEM_PROCESSOR} ")" )
option(ON_VISIONGPU1 "Option description" ON)
add_definitions(-DON_VISIONGPU1)

set(CMAKE_CXX_FLAGS "-std=c++11")

set(EXECUTABLE_OUTPUT_PATH ${MY_DIR}/bin)

# OpenCV
set(OpenCV_DIR /home/linguangzhang/software/opencv-2.4.13/install/share/OpenCV)
include(${OpenCV_DIR}/OpenCVConfig.cmake)
include(${OpenCV_DIR}/OpenCVModules.cmake)
message(STATUS "OpenCV_DIR=" ${OpenCV_DIR})


# Eigen
set(Eigen3_INCLUDE_DIR /usr/include/eigen3)
# find_package(Eigen3 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIR})

# FLANN
# set(Flann_INCLUDE_DIR /usr/local/include/flann)
# set(Flann_LIBS_DIR /usr/local/lib)
set(Flann_INCLUDE_DIR /home/linguangzhang/software/flann/install/include)
set(Flann_LIBS_DIR /home/linguangzhang/software/flann/install/lib)
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
set(SiftGPU_INCLUDE_DIR /home/linguangzhang/software/SiftGPU/src/SiftGPU)
set(SiftGPU_LIBS_DIR /home/linguangzhang/software/SiftGPU/bin)
include_directories(${SiftGPU_INCLUDE_DIR})
link_directories(${SiftGPU_LIBS_DIR})

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib)


# GFLAGS
# find_package(gflags REQUIRED)
# message(STATUS "gflags_include = " ${gflags_INCLUDE_DIR})
# message(STATUS ${gflags_LIBS_DIR})
include_directories(/home/linguangzhang/software/gflags/build/include)
 link_directories(/home/linguangzhang/software/gflags/build/lib)
