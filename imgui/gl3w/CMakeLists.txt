set(SRCS
  GL/gl3w.c
)

add_library(gl3w ${SRCS})

if(ON_MAC)
  target_link_libraries(gl3w)
elseif(ON_TEGRA OR ON_LINUX)
  target_link_libraries(gl3w
        dl
        rt 
        m 
        Xrandr 
        Xinerama 
        Xi 
        Xcursor 
        Xrender 
        GL 
        pthread 
        dl 
        drm 
        Xdamage 
        Xfixes 
        X11-xcb 
        xcb-glx 
        xcb-dri2 
        Xxf86vm 
        Xext 
        X11 
        xcb 
        Xau 
        Xdmcp)
endif()
