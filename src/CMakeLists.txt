set(SRCS
  # main.cpp
  work_image.cpp
  database.cpp
  micro_gps.cpp
  inpolygon.c
)

add_library(mgps ${SRCS})


if(ON_MAC)
  target_link_libraries(mgps
        ${OpenCV_LIBS}
        flann_cuda_s
        # flann
        cudart
        siftgpu
        )
        
elseif(ON_CSCYCLE)
  target_link_libraries(mgps
        ${OpenCV_LIBS}
        flann
        cudart
        siftgpu
        )

elseif(ON_TEGRA)
  target_link_libraries(mgps
        ${OpenCV_LIBS}
        flann
        cudart
        siftgpu
        )
endif()
