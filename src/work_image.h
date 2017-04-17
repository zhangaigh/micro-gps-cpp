#ifndef _WORK_IMAGE_H_
#define _WORK_IMAGE_H_

// C/C++
#include "stdio.h"
#include <vector>
#include <iostream>
#include <algorithm>

// Opencv2.4
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/imgproc/imgproc.hpp>

// Eigen3
#include <Eigen/Dense>

#define USE_SIFT_GPU

// SiftGPU
#ifdef USE_SIFT_GPU
#include "SiftGPU.h"
void initSiftGPU();
#endif

struct SIFTFeature {
  ~SIFTFeature() {

  }

  float x = -1.0f;
  float y = -1.0f;
  float scale = -1.0f;
  float angle = -1.0f;
  float strength = 0.0f;
  // float descriptor[128];
  Eigen::RowVectorXf descriptor;
  Eigen::RowVectorXf descriptor_PCAed;
  Eigen::Matrix3f local_pose;
  Eigen::Matrix3f global_pose;
};


class WorkImage {
public:
  // TODO: improve constructor and destructor
  WorkImage(const char* image_path, const char* precomputed_feature_path = NULL); 
  WorkImage(int width, int height, bool is_color=true);
  ~WorkImage();
  void release();

  void loadImage();
  void toGray();
  void resize(float ratio);
  void extractSIFT(float downsampling = 0.5f);
  void saveSIFTFeatures(char* path, int num_samples = 9999, float margin_thresh = -1.0f);
  bool loadPrecomputedFeature(float downsampling = 0.5f); // can be anything
  void PCADimReduction(Eigen::MatrixXf& PCA_basis);

  // access functions
  char* getImagePath();
  int width();
  int height();
  int channels();
  uchar* data();
  size_t getSIFTSize();
  SIFTFeature* getSIFTFeature(int idx);
  void write(const char* image_path);


  // matching functions
  void siftMatch(WorkImage* img,
                std::vector<int>& matched_idx1,
                std::vector<int>& matched_idx2);

  bool siftMatchEstimatePose(WorkImage* img,
                std::vector<int>& matched_idx1,
                std::vector<int>& matched_idx2,
                Eigen::MatrixXf& pose,
                std::vector<int>& inliers);

private:
  char m_image_path[256];
  bool m_is_color;
  uchar* m_data;

  int m_width;
  int m_height;
  float sift_extraction_scale;

  cv::SIFT* m_sift_detector;
  std::vector<SIFTFeature*> m_sift_features;
  // SiftGPU* m_sift_gpu;

  char m_external_feature_path[256];

};


inline size_t WorkImage::getSIFTSize() {
  return m_sift_features.size();
}

inline SIFTFeature* WorkImage::getSIFTFeature(int idx) {
  return m_sift_features[idx];
}

inline char* WorkImage::getImagePath() {
  return m_image_path;
}

inline int WorkImage::width() {
  return m_width;
}

inline int WorkImage::height() {
  return m_height;
}

inline int WorkImage::channels() {
  if (m_is_color) {
    return 3;
  } else {
    return 1;
  }
}

inline uchar* WorkImage::data() {
  return m_data;
}

void computeRigidTransformation(Eigen::MatrixXf points1, Eigen::MatrixXf points2,
                                  Eigen::MatrixXf& pose);



void estimateRigidTransformation(Eigen::MatrixXf points1, Eigen::MatrixXf points2,
                                  Eigen::MatrixXf& pose, std::vector<int>& inliers,
                                  int num_iterations = 1000,
                                  float error_thresh = 3.0f); // 3 pixels


void computeImageArrayWorldSize(std::vector<WorkImage*>& images,
                                std::vector<Eigen::Matrix3f>& poses,
                                float warp_scale,
                                int& world_min_x,
                                int& world_min_y,
                                int& world_max_x,
                                int& world_max_y);

WorkImage* warpImageArray(std::vector<WorkImage*>& images,
                        std::vector<Eigen::Matrix3f>& poses,
                        float warp_scale = 0.25);
                        
// void warpImageArray(std::vector<WorkImage*> images,
//                     std::vector<Eigen::Matrix3f> poses,
//                     WorkImage*& output_image,
//                     float warp_scale = 0.25);

#endif
