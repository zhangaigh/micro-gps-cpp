set(SRCS
    image_dataset.cpp
    image.cpp
    image_func.cpp
    micro_gps.cpp
    inpolygon.c)

add_library(mgps ${SRCS})

if(ON_MAC)
  target_link_libraries(mgps
        ${OpenCV_LIBS}
        flann_cuda_s
        # flann
        cudart
        siftgpu)


  # add_executable(test_image_dataset image_dataset.cpp)
  # target_link_libraries(test_image_dataset)

  # add_executable(test_image_basics test_image_basics.cpp
  #                                  image_dataset.cpp
  #                                  image.cpp
  #                                  image_func.cpp
  #                                  micro_gps2.cpp
  #                                  inpolygon2.c)
                                   
  # target_link_libraries(test_image_basics
  #                       ${OpenCV_LIBS}
  #                       siftgpu)


elseif(ON_VISIONGPU1)
  target_link_libraries(mgps
        ${OpenCV_LIBS}
        flann
        # cudart
        siftgpu)

        
elseif(ON_TEGRA)
  target_link_libraries(mgps
        ${OpenCV_LIBS}
        flann
        # cudart
        siftgpu)

endif()
