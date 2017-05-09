#ifndef _IMAGE_FUNC_H_ 
#define _IMAGE_FUNC_H_

#include "flann/flann.h"
#include "image.h"
#include "util.h"


namespace MicroGPS {
namespace ImageFunc {


Image*    cropPatch(const Image* im,
                    const float x, const float y, const float orientation,
                    const int win_width, const int win_height);


void      matchFeatureBidirectional(Image* img1_ptr, Image* img2_ptr,
                                    std::vector<int>& matched_idx1,
                                    std::vector<int>& matched_idx2,
                                    bool extract_feature=true);


// compute rigid transformation using matched keypoints, used by RANSAC
void      computeRigidTransformation(Eigen::MatrixXf points1, Eigen::MatrixXf points2,
                                     Eigen::MatrixXf& pose);


// RANSAC - estimate rigid transformation
void      estimateRigidTransformationRANSAC(Eigen::MatrixXf points1, Eigen::MatrixXf points2,
                                            Eigen::MatrixXf& pose, std::vector<int>& inliers,
                                            int num_iterations = 1000,
                                            float error_thresh = 3.0f); // 3 pixels


// use matched features and estimate pose by RANSAC
bool      estimatePoseFromMatchedImages(Image* img1_ptr, Image* img2_ptr,
                                        std::vector<int>& matched_idx1,
                                        std::vector<int>& matched_idx2,
                                        Eigen::MatrixXf& pose,
                                        std::vector<int>& inliers,
                                        int num_ransac_iterations = 1000,
                                        float error_thresh = 3.0f);

// compute the world limits for an image array
void      computeImageArrayWorldLimits(std::vector<Image*>& images,
                                       std::vector<Eigen::Matrix3f>& poses,
                                       float warp_scale,
                                       int& world_min_x,
                                       int& world_min_y,
                                       int& world_max_x,
                                       int& world_max_y);

// precompute warp mapping, do not do actual warping
void      computeWarpImageMapping(int im_width, int im_height,
                                  int world_min_x, int world_min_y,
                                  int world_max_x, int world_max_y,
                                  Eigen::Matrix3f pose,
                                  std::vector<Eigen::Vector2f>& world_location,
                                  std::vector<Eigen::Vector2f>& im_location);

// warp an image array to generate a map
Image*    warpImageArray(std::vector<Image*>& images,
                         std::vector<Eigen::Matrix3f>& poses,
                         float warp_scale = 0.25);




}
}


#endif