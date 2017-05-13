# set(CMAKE_CXX_COMPILER clang-omp++)
# set(CMAKE_C_COMPILER clang-omp)


message( STATUS "Building for:  " ${CMAKE_SYSTEM} "  FlyCapture not available")
option(ON_MAC "Option description" ON)
add_definitions(-DON_MAC)
add_definitions(-DGFLAGS_NAMESPACE=gflags)


set(CMAKE_CXX_FLAGS "-std=c++11 -mmacosx-version-min=10.9 -framework OpenGL -framework Cocoa")

set(EXECUTABLE_OUTPUT_PATH ${MY_DIR}/bin_mac)


# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})


# Eigen
set(Eigen3_INCLUDE_DIR /usr/local/include/eigen3)
# find_package(Eigen3 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIR})

# FLANN
# set(Flann_INCLUDE_DIR /usr/local/include/flann)
# set(Flann_LIBS_DIR /usr/local/lib)
set(Flann_INCLUDE_DIR ~/software/flann-1.8.4-src/install/include)
set(Flann_LIBS_DIR ~/software/flann-1.8.4-src/install/lib)
include_directories(${Flann_INCLUDE_DIR})
link_directories(${Flann_LIBS_DIR})

# GLFW
find_package(GLFW REQUIRED)
include_directories(${GLFW_INCLUDE_DIR})
link_directories(${GLFW_LIBS_DIR})

# # VLFEAT
# set(VLFEAT_INCLUDE_DIR ${MY_DIR}/vlfeat)
# set(VLFEAT_LIBS_DIR ${MY_DIR}/vlfeat/bin/maci64)
# include_directories(${VLFEAT_INCLUDE_DIR})
# link_directories(${VLFEAT_LIBS_DIR})

# SiftGPU
set(SiftGPU_INCLUDE_DIR ~/software/SiftGPU/src/SiftGPU)
set(SiftGPU_LIBS_DIR ~/software/SiftGPU/bin)
include_directories(${SiftGPU_INCLUDE_DIR})
link_directories(${SiftGPU_LIBS_DIR})

# CUDA
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib)

# GFLAGS
find_package(gflags REQUIRED)
include_directories(${gflags_INCLUDE_DIR})
link_directories(${gflags_LIBS_DIR})
