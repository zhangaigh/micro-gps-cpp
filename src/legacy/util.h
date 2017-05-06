#ifndef _UTIL_H_
#define _UTIL_H_

#include <ctime>
#include <cstdlib>

#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>
#include <Eigen/Dense>
#include <iostream>

// helpers
inline int myrandom (int i) {
  static bool initialized = false;
  if (!initialized) {
    std::srand(unsigned(std::time(0)));
    initialized = true;
  }  
  return std::rand()%i;
}

inline void randomSample(int n, int k, std::vector<int>& sel) {
  if (k > n) {
    k = n;
  }

  sel.resize(n);
  for (int i = 0; i < n; i++) {
    sel[i] = i;
  }

  std::random_shuffle (sel.begin(), sel.end(), myrandom);
  sel.resize(k);
}


inline void readRobotCameraCalibration (char* file_path) {
  Eigen::MatrixXf T_camera_robot = Eigen::MatrixXf::Identity(3, 3);
  float mm_per_pixel;
  FILE* fp = fopen(file_path, "r");
  fscanf(fp, "%f %f %f\n", &T_camera_robot(0, 0), &T_camera_robot(0, 1), &T_camera_robot(0, 2));
  fscanf(fp, "%f %f %f\n", &T_camera_robot(1, 0), &T_camera_robot(1, 1), &T_camera_robot(1, 2));
  fscanf(fp, "%f\n", &mm_per_pixel);
  fclose(fp);

  std::cout << "T_camera_robot = " << T_camera_robot << std::endl;
  std::cout << "mm_per_pixel = " << mm_per_pixel << std::endl;
}


inline int listDir(std::string dir, std::vector<std::string>& files, std::string keyword, bool fname_only = false) {
  files.clear();

  DIR* dp;
  struct dirent* dirp;
  if((dp  = opendir(dir.c_str())) == NULL) {
      return -1;
  }

  while ((dirp = readdir(dp)) != NULL) {
  	std::string name = std::string(dirp->d_name);
    // exceptions
  	if(name != "." && name != ".." && name != ".DS_Store") {
      if (keyword != ""){
        if (name.find(keyword) != std::string::npos) {
          files.push_back(name);
        }
      } else {
        files.push_back(name);
      }
    }
  }
  closedir(dp);

  // sort files
  std::sort(files.begin(), files.end());

  if(dir.at( dir.length() - 1 ) != '/') {
    dir = dir + "/";
  }

  for(unsigned int i = 0;i<files.size();i++) {
    if(files[i].at(0) != '/') {
      if (!fname_only) {
        files[i] = dir + files[i];
      }
    }
  }

  return files.size();
}


inline bool checkFileExists(const char* path) {
  struct stat buffer;   
  if (stat (path, &buffer) != 0) {
    return false;
  }
  return true;
}

inline void mkdirIfNotExists(char* path) {
  struct stat buffer;   
  if (stat (path, &buffer) != 0) {
    printf("%s doesn't exist!\n", path);
    char cmd[256];
    sprintf(cmd, "mkdir %s", path);
    system(cmd);
  } else {
    printf("%s exists, ignore...\n", path);
  }
}


#endif
