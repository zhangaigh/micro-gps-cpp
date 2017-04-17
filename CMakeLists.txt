cmake_minimum_required (VERSION 2.6)

project(keypoint)
set(CMAKE_BUILD_TYPE Release)

set(MY_DIR ${keypoint_SOURCE_DIR})

message(STATUS ${CMAKE_SYSTEM_PROCESSOR} "   " ${CMAKE_SYSTEM})

include(config.cmake)

add_subdirectory(imgui_lib)
include_directories(imgui_lib)

add_subdirectory(src)
include_directories(src)

add_subdirectory(exec)
