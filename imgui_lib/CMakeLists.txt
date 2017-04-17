set(SRCS
  imgui.cpp
  imgui_draw.cpp
  imgui_demo.cpp
)

add_library(imgui ${SRCS})
target_link_libraries(imgui)
