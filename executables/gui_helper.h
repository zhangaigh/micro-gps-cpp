#ifndef _GUI_HELPER_H_
#define _GUI_HELPER_H_

#include <imgui.h>
#include "micro_gps.h"
#ifdef ON_MAC
#include "OpenGL/gl.h"
#else
#include "GL/gl.h"
#endif


struct GLFWDisplay {
  int screen_w;
  int screen_h;
  int framebuffer_w;
  int framebuffer_h;
};


struct ImageGL3Texture {
  GLuint  id;
  float   raw_width; // original texture width (no matter rotated or not)
  float   raw_height;
  float   width;
  float   height;
  bool    rotated90;
  bool    active;

  void  loadTextureFromImage(MicroGPS::Image* image, bool rotate90 = false);
  // void loadTextureFromCVMat(cv::Mat& image, bool rotate90 = false);
  void  deactivate();
  // float width(); // get (rotated) width
  // float height(); 

  ImageGL3Texture();
};


static inline void gui_error_callback(int error, const char* description) {
  fprintf(stderr, "Error %d: %s\n", error, description);
}


namespace ImGui {
  static inline bool Combo(const char* label, 
                           int* current_item, 
                           const std::vector<std::string>& items, 
                           int height_in_items = -1) {
    return Combo(label, current_item,
                 [](void* data, int idx, const char** out_text) {
                     *out_text = (*(const std::vector<std::string>*)data)[idx].c_str();
                     return true;
                   },
                 (void*)&items, items.size(), height_in_items);

  }
}

#endif