#ifndef _GUI_HELPER_H_
#define _GUI_HELPER_H_

#include "micro_gps.h"

struct WorkImageGL3Texture {
  GLuint id;
  float width; // original texture width (no matter rotated or not)
  float height;
  bool show;
  bool rotated90;

  void loadTextureFromWorkImage(WorkImage* work_image, bool rotate90 = false);
  void loadTextureFromCVMat(cv::Mat& image, bool rotate90 = false);
  void disable();
  float getWidth(); // get (rotated) width
  float getHeight(); 

  WorkImageGL3Texture();
};


struct GLFWDisplay {
  int screen_w;
  int screen_h;
  int framebuffer_w;
  int framebuffer_h;
};

struct RenderedTextureInfo {
  float screen_pos_x;
  float screen_pos_y;
  float fitting_scale;
  float width;
  float height;
  bool rotated90;
};

#endif