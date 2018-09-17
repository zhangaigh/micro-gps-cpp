# include_directories(gl3w)
# add_subdirectory(gl3w)

# set(GUI_SRCS
#   # imgui_impl_glfw_gl3.cpp
#   # gui.cpp
#   # micro_gps_exec.cpp
# )


if(ON_MAC)
  # add_executable(test_image_basics ${GUI_SRCS})
  add_executable(micro_gps_exec micro_gps_exec.cpp)
  add_executable(batch_extract_sift batch_extract_sift.cpp)
  add_executable(batch_crop_sift_patches batch_crop_sift_patches.cpp)
  add_executable(micro_gps_gui micro_gps_gui.cpp
                               gui_helper.cpp)

  target_link_libraries(micro_gps_exec
                        mgps
                        gflags)

  target_link_libraries(batch_extract_sift
                        mgps
                        gflags)

  target_link_libraries(batch_crop_sift_patches
                        mgps
                        gflags)

  target_link_libraries(micro_gps_gui
                        mgps
                        imgui_impl
                        gflags)
# --------------------------------------------------------------------

elseif(ON_TEGRA)

# --------------------------------------------------------------------

elseif(ON_LINUX)
  add_executable(micro_gps_exec micro_gps_exec.cpp)
  add_executable(batch_extract_sift batch_extract_sift.cpp)
  add_executable(batch_crop_sift_patches batch_crop_sift_patches.cpp)
  add_executable(test test.cpp)
  
  target_link_libraries(micro_gps_exec
                        mgps
                        gflags)

  target_link_libraries(batch_extract_sift
                        mgps
                        gflags)

  target_link_libraries(batch_crop_sift_patches
                        mgps
                        gflags)

  target_link_libraries(test
                        gflags)


  # add_executable(batch_extract_sift batch_extract_sift.cpp)
  # target_link_libraries(batch_extract_sift
  #                       mgps
  #                       gflags)

endif()
