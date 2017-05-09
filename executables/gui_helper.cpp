#include "gui_helper.h"


ImageGL3Texture::ImageGL3Texture() {
  id = -1;
  raw_width = -1.0f;
  raw_height = -1.0f;
  width = -1.0f;
  height = -1.0f;
  active = false;
}

void ImageGL3Texture::loadTextureFromImage(MicroGPS::Image* image, bool rotate90) {
  if (active) {
    this->deactivate();
  }

  raw_width = (float)image->width(); // store the origin size
  raw_height = (float)image->height();

  rotated90 = rotate90;
  if (rotate90) {
    image->rotate90(); // warning: changes the original image
    width = raw_height;
    height = raw_width;
  } else {
    width = raw_width;
    height = raw_height;
  }

  glGenTextures(1, &id);

  glBindTexture(GL_TEXTURE_2D, id);

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // by default number of bytes in a row should be multiples of 4, causing problems
  // upload the image to OpenGL
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 
              image->width(), image->height(), 
              0, GL_BGR, GL_UNSIGNED_BYTE, image->data());

  printf("uploaded GL texture\n");

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glGenerateMipmap(GL_TEXTURE_2D);

  active = true; // ready to display
}

void ImageGL3Texture::deactivate() {
  if (active) { // current being displayed
    glDeleteTextures(1, &id);
  }
  id = -1;
  raw_width = -1.0f;
  raw_height = -1.0f;
  width = -1.0f;
  height = -1.0f;
  active = false;
}
