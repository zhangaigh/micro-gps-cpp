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
  add_executable(micro_gps_gui micro_gps_gui.cpp)

  target_link_libraries(micro_gps_exec
                        mgps
                        gflags)

  target_link_libraries(batch_extract_sift
                        mgps
                        gflags)

  target_link_libraries(micro_gps_gui
                        mgps
                        imgui_impl
                        gflags)

  # set(ONLINE_GUI_SRCS
  #   imgui_impl_glfw_gl3.cpp
  #   online_gui.cpp
  #   robot.cpp
  #   gui_helper.cpp)

  # add_executable(online_gui ${ONLINE_GUI_SRCS})
  # target_link_libraries(online_gui
  #                       glfw3
  #                       gl3w
  #                       imgui
                        # mgps)


  # set(MONITOR_SRCS
  #   imgui_impl_glfw_gl3.cpp
  #   monitor.cpp
  #   robot.cpp
  #   gui_helper.cpp)

  # add_executable(monitor ${MONITOR_SRCS})
  # target_link_libraries(monitor
  #                       glfw3
  #                       gl3w
  #                       imgui
  #                       mgps)


  # add_executable(calibrate_robot_camera calibrate_robot_camera.cpp)
  # target_link_libraries(calibrate_robot_camera
  #                       ${OpenCV_LIBS})


  # add_executable(batch_extract_sift batch_extract_sift.cpp)
  # target_link_libraries(batch_extract_sift
  #                       mgps
  #                       gflags)

  # add_executable(expand_map expand_map.cpp
  #                           refine_poses.cpp)
  # target_link_libraries(expand_map
  #                       mgps
  #                       gflags
  #                       ceres
  #                       glog)
# --------------------------------------------------------------------

elseif(ON_TEGRA)
  # add_executable(opengl3_gui ${GUI_SRCS})
  # target_link_libraries(opengl3_gui
  #       /usr/lib/aarch64-linux-gnu/libglfw.so.3
  #       GL
  #       glfw
  #       gl3w
  #       imgui
  #       mgps
  #       /usr/lib/aarch64-linux-gnu/libgflags.so)

  # set(ONLINE_GUI_SRCS
  #   imgui_impl_glfw_gl3.cpp
  #   online_gui.cpp
  #   robot.cpp
  #   gui_helper.cpp
  #   flycapture_wrapper.cpp
  #   calibrate_intrinsics.cpp)

  # add_executable(online_gui ${ONLINE_GUI_SRCS})
  # target_link_libraries(online_gui
  #                       /usr/lib/aarch64-linux-gnu/libglfw.so.3
  #                       GL
  #                       glfw
  #                       gl3w
  #                       imgui
  #                       mgps
  #                       flycapture)
# --------------------------------------------------------------------

elseif(ON_VISIONGPU1)
  # add_executable(opengl3_gui ${GUI_SRCS})
  # target_link_libraries(opengl3_gui
  #                       glfw3
  #                       gl3w
  #                       imgui
  #                       mgps
  #                       gflags)

  # add_executable(batch_extract_sift batch_extract_sift.cpp)
  # target_link_libraries(batch_extract_sift
  #                       mgps
  #                       gflags)

endif()
