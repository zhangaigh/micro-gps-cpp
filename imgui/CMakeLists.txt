# include_directories(gl3w)
add_subdirectory(gl3w)

# include_directories(lib)
add_subdirectory(lib)

add_library(imgui_impl imgui_impl_glfw_gl3.cpp)


if(ON_MAC)
  target_link_libraries(imgui_impl
                        glfw3
                        gl3w
                        imgui)
# --------------------------------------------------------------------
elseif(ON_TEGRA)
  target_link_libraries(imgui_impl
                        /usr/lib/aarch64-linux-gnu/libglfw.so.3
                        GL
                        glfw
                        gl3w
                        imgui)
# --------------------------------------------------------------------
elseif(ON_VISIONGPU1)
  target_link_libraries(imgui_impl
                        glfw3
                        gl3w
                        imgui)
endif()
