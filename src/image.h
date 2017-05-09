#ifndef _IMAGE_H_
#define _IMAGE_H_

// C/C++
#include "stdio.h"
#include <vector>
#include <iostream>
#include <algorithm>

// Opencv2.4
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/nonfree/features2d.hpp"
//#include "opencv2/nonfree/nonfree.hpp"

// Eigen3
#include <Eigen/Dense>



namespace MicroGPS {

void initSiftGPU();

struct LocalFeature {
  ~LocalFeature() {

  }

  float               x = -1.0f;
  float               y = -1.0f;
  float               scale = -1.0f;
  float               angle = -1.0f;
  float               strength = 0.0f;
  Eigen::RowVectorXf  descriptor;
  Eigen::RowVectorXf  descriptor_compressed;
  Eigen::Matrix3f     local_pose;
  Eigen::Matrix3f     global_pose;
};


class Image {

public:
  // constructors
  Image();
  Image(const char* image_path); 
  Image(const char* image_path, 
        const char* precomputed_feat_path, 
        const char* precomputed_sift_path);
  Image(const size_t width, const size_t height, const size_t channels);
  
  // destructor
  ~Image();

  // load / release buffer
  void            loadImage();
  void            create(const size_t width, const size_t height, const size_t channels); 
  void            release();   

  // access functions
  const char*     getImagePath() const;
  size_t          width() const;
  size_t          height() const;
  size_t          channels() const;
  uchar*          data() const;
  uchar&          getPixel(unsigned row, unsigned col, unsigned ch = 0) const;
  uchar&          operator() (unsigned row, unsigned col, unsigned ch = 0) const;
  void            write(const char* image_path) const;
  void            show(const char* win_name) const;

  // basic processing
  float           bilinearSample (float y, float x, unsigned ch = 0) const;
  void            bgr2gray();
  void            resize(const float ratio);
  void            rotate90(bool clockwise = true);
  void            flip(bool horizontal = false);
  Image*          clone() const;

  // crop a patch
  Image*          cropPatch(const float x, const float y, const float orientation,
                            const int win_width, const int win_height) const;

  // Features
  void            extractSIFT(float downsampling = 0.5f);
  void            saveLocalFeatures(const char* path, 
                                    const int num_samples = 9999, 
                                    const float margin_thresh = -1.0f);
  bool            loadPrecomputedFeatures(const bool load_sift);
  size_t          getNumLocalFeatures();
  LocalFeature*   getLocalFeature(size_t idx);
  void            linearFeatureCompression(const Eigen::MatrixXf& basis);

  // helper
  cv::Mat         convertToCvMat() const;
private:
  
  char                        m_image_path[256];
  char                        m_precomputed_feat_path[256];
  char                        m_precomputed_sift_path[256];
  uchar*                      m_data;

  size_t                      m_width;
  size_t                      m_height;
  size_t                      m_channels;

  std::vector<LocalFeature*>  m_local_features;

};



}

#endif
