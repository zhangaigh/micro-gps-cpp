cmake_minimum_required (VERSION 2.6)

project(micro_gps)
set(CMAKE_BUILD_TYPE Release)

set(MY_DIR ${micro_gps_SOURCE_DIR})

message(STATUS ${CMAKE_SYSTEM_PROCESSOR} "   " ${CMAKE_SYSTEM})

include(config.cmake)

if (NOT VISIONGPU1)
  include_directories(imgui)
  include_directories(imgui/gl3w)
  include_directories(imgui/lib)
  add_subdirectory(imgui)
endif()

include_directories(src)
add_subdirectory(src)

add_subdirectory(executables)
