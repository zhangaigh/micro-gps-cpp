#include <imgui.h>
#include "imgui_impl_glfw_gl3.h"
#include <GL/gl3w.h>
#include <GLFW/glfw3.h>
#include <fstream>
#include <vector>
#include "gui_helper.h"
#include "dirent.h"
#include <stdio.h>

// int listDir(std::string dir, std::vector<std::string>& files, std::string keyword, bool fname_only = false) {
//   files.clear();

//   DIR* dp;
//   struct dirent* dirp;
//   if((dp  = opendir(dir.c_str())) == NULL) {
//       return -1;
//   }

//   while ((dirp = readdir(dp)) != NULL) {
//   	std::string name = std::string(dirp->d_name);
//     // exceptions
//   	if(name != "." && name != ".." && name != ".DS_Store") {
//       if (keyword != ""){
//         if (name.find(keyword) != std::string::npos) {
//           files.push_back(name);
//         }
//       } else {
//         files.push_back(name);
//       }
//     }
//   }
//   closedir(dp);

//   // sort files
//   std::sort(files.begin(), files.end());

//   if(dir.at( dir.length() - 1 ) != '/') {
//     dir = dir + "/";
//   }

//   for(unsigned int i = 0;i<files.size();i++) {
//     if(files[i].at(0) != '/') {
//       if (!fname_only) {
//         files[i] = dir + files[i];
//       }
//     }
//   }

//   return files.size();
// }


WorkImageGL3Texture::WorkImageGL3Texture() {
  id = -1;
  width = -1.0f;
  height = -1.0f;
  show = false;
}

void WorkImageGL3Texture::loadTextureFromWorkImage(WorkImage* work_image, bool rotate90) {
  cv::Mat cvImg;
  if (work_image->channels() == 3) {
    cvImg = cv::Mat(work_image->height(), work_image->width(), CV_8UC3, work_image->data());
  } else {
    cvImg = cv::Mat(work_image->height(), work_image->width(), CV_8UC1, work_image->data());
    cv::cvtColor(cvImg, cvImg, CV_GRAY2BGR);
  }
  this->loadTextureFromCVMat(cvImg, rotate90);
}

void WorkImageGL3Texture::loadTextureFromCVMat(cv::Mat& image, bool rotate90) {
  if (show) {
    this->disable();
  }

  cv::Mat image_to_load = image;
  rotated90 = rotate90;
  if (rotated90) {
    cv::transpose(image, image_to_load);
    cv::flip(image_to_load, image_to_load, 1);
  }

  glGenTextures(1, &id);

  glBindTexture(GL_TEXTURE_2D, id);

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // by default number of bytes in a row should be multiples of 4, causing problems
  // upload the image to OpenGL
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image_to_load.cols, image_to_load.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, image_to_load.data);

  printf("uploaded image\n");

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glGenerateMipmap(GL_TEXTURE_2D);


  width = image.cols; // still keep the origin size
  height = image.rows;

  show = true; // ready to display
}


void WorkImageGL3Texture::disable() {
  if (show) { // current being displayed
    glDeleteTextures(1, &id);
  }
  id = -1;
  width = -1.0f;
  height = -1.0f;
  show = false;
}

float WorkImageGL3Texture::getWidth() {
  if (rotated90) {
    return height;
  }
  return width;
}

float WorkImageGL3Texture::getHeight() {
  if (rotated90) {
    return width;
  }
  return height;
}
