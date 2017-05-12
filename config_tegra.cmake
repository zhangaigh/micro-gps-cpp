message( STATUS "Building for:  " ${CMAKE_SYSTEM} "(" ${CMAKE_SYSTEM_PROCESSOR} ")" )
option(ON_TEGRA "Option description" ON)
add_definitions(-DON_TEGRA)


set(CMAKE_CXX_FLAGS "-std=c++11")

set(EXECUTABLE_OUTPUT_PATH ${MY_DIR}/bin)


# OpenCV

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})


# Eigen
set(Eigen3_INCLUDE_DIR /usr/include/eigen3)
# find_package(Eigen3 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIR})

# FLANN
set(Flann_INCLUDE_DIR /usr/include/flann)
set(Flann_LIBS_DIR /usr/lib)
include_directories(${Flann_INCLUDE_DIR})
# link_directories(${Flann_LIBS_DIR})

# GLFW
set(GLFW_INCLUDE_FIR /usr/include)
set(GLFW_LIBS_DIR /usr/lib/aarch64-linux-gnu)
#find_package(GLFW REQUIRED)
include_directories(${GLFW_INCLUDE_DIR})
#link_directories(${GLFW_LIBS_DIR})

# # VLFEAT
# set(VLFEAT_INCLUDE_DIR /home/ubuntu/mgps-thirdparty/vlfeat)
# set(VLFEAT_LIBS_DIR /home/ubuntu/mgps-thirdparty/vlfeat/bin/aarch64)
# include_directories(${VLFEAT_INCLUDE_DIR})
# link_directories(${VLFEAT_LIBS_DIR})

# Flycapture
set(FlyCapture_INCLUDE_DIR /home/ubuntu/Documents/libflycapture/include)
set(FlyCapture_LIBS_DIR /home/ubuntu/Documents/libflycapture/lib)
include_directories(${FlyCapture_INCLUDE_DIR})
link_directories(${FlyCapture_LIBS_DIR})

# SiftGPU
set(SiftGPU_INCLUDE_DIR /home/ubuntu/mgps-thirdparty//SiftGPU/src/SiftGPU)
set(SiftGPU_LIBS_DIR /home/ubuntu/mgps-thirdparty/SiftGPU/bin)
include_directories(${SiftGPU_INCLUDE_DIR})
link_directories(${SiftGPU_LIBS_DIR})

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# GFLAGS
# find_package(gflags REQUIRED)
include_directories(/usr/include/gflags)
link_directories(/usr/lib/aarch64-linux-gnu)
