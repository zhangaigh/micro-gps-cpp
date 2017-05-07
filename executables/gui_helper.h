#ifndef _GUI_HELPER_H_
#define _GUI_HELPER_H_


struct GLFWDisplay {
  int screen_w;
  int screen_h;
  int framebuffer_w;
  int framebuffer_h;
};



static inline void gui_error_callback(int error, const char* description) {
  fprintf(stderr, "Error %d: %s\n", error, description);
}


#endif