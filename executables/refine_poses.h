#ifndef _REFINE_POSES_H_
#define _REFINE_POSES_H_
#include "ceres/ceres.h"
#include "glog/logging.h"

// #include "math.h"

template <typename T>
void invPose (const T* const pose_in, T* pose_out) {
  pose_out[0] = -pose_in[0];
  pose_out[1] = - (cos(pose_in[0]) * pose_in[1] + sin(pose_in[0]) * pose_in[2]);
  pose_out[2] =  - (-sin(pose_in[0]) * pose_in[1] + cos(pose_in[0]) * pose_in[2]);
}

template <typename T>
void applyPose (T x, T y, const T* const pose, T& x_, T& y_) {
  x_ = cos(pose[0]) * x - sin(pose[0]) * y + pose[1];
  y_ = sin(pose[0]) * x + cos(pose[0]) * y + pose[2];
}


// Rigid transformation
template <typename T>
void invPoseRigid (const T* const pose_in, T* pose_out) {
  pose_out[0] = -pose_in[0];
  pose_out[1] = - (cos(pose_in[0]) * pose_in[1] + sin(pose_in[0]) * pose_in[2]);
  pose_out[2] =  - (-sin(pose_in[0]) * pose_in[1] + cos(pose_in[0]) * pose_in[2]);
}

template <typename T>
void applyPoseRigid (T x, T y, const T* const pose, T& x_, T& y_) {
  x_ = cos(pose[0]) * x - sin(pose[0]) * y + pose[1];
  y_ = sin(pose[0]) * x + cos(pose[0]) * y + pose[2];
}


template <typename T>
void invPoseSimilarity (const T* const pose_in, T* pose_out) {
  pose_out[0] = -pose_in[0];
  pose_out[1] = T(1.0f) / pose_in[1];
  pose_out[2] = - (cos(pose_in[0]) * pose_in[2] + sin(pose_in[0]) * pose_in[3]) / pose_in[1];
  pose_out[3] =  - (-sin(pose_in[0]) * pose_in[2] + cos(pose_in[0]) * pose_in[3]) / pose_in[1];
}

template <typename T>
void applyPoseSimilarity (T x, T y, const T* const pose, T& x_, T& y_) {
  x_ = (cos(pose[0]) * x - sin(pose[0]) * y) * pose[1] + pose[2];
  y_ = (sin(pose[0]) * x + cos(pose[0]) * y) * pose[1] + pose[3];
}

// affine inverse
// [  e/(a*e - b*d), -b/(a*e - b*d),  (b*f - c*e)/(a*e - b*d)]
// [ -d/(a*e - b*d),  a/(a*e - b*d), -(a*f - c*d)/(a*e - b*d)]
// [              0,              0,                        1]

template <typename T>
void invPose6DOF (const T* const pose_in, T* pose_out) {
  T a = pose_in[0];
  T b = pose_in[1];
  T c = pose_in[2];
  T d = pose_in[3];
  T e = pose_in[4];
  T f = pose_in[5];

  pose_out[0] = e/(a*e - b*d);
  pose_out[1] = -b/(a*e - b*d);
  pose_out[2] = (b*f - c*e)/(a*e - b*d);
  pose_out[3] = -d/(a*e - b*d);
  pose_out[4] = a/(a*e - b*d);
  pose_out[5] = -(a*f - c*d)/(a*e - b*d);
}

template <typename T>
void applyPose6DOF (T x, T y, const T* const pose, T& x_, T& y_) {
  x_ = pose[0] * x + pose[1] * y + pose[2];
  y_ = pose[3] * x + pose[4] * y + pose[5];
}


template <typename T>
void invPose8DOF (const T* const pose_in, T* pose_out) {
  T a = pose_in[0];
  T b = pose_in[1];
  T c = pose_in[2];
  T d = pose_in[3];
  T e = pose_in[4];
  T f = pose_in[5];
  T g = pose_in[6];
  T h = pose_in[7];

  pose_out[0] = (e - f*h)/(a*e - b*d - a*f*h + b*f*g + c*d*h - c*e*g);
  pose_out[1] = -(b - c*h)/(a*e - b*d - a*f*h + b*f*g + c*d*h - c*e*g);
  pose_out[2] = (b*f - c*e)/(a*e - b*d - a*f*h + b*f*g + c*d*h - c*e*g);
  pose_out[3] = -(d - f*g)/(a*e - b*d - a*f*h + b*f*g + c*d*h - c*e*g);
  pose_out[4] = (a - c*g)/(a*e - b*d - a*f*h + b*f*g + c*d*h - c*e*g);
  pose_out[5] = -(a*f - c*d)/(a*e - b*d - a*f*h + b*f*g + c*d*h - c*e*g);
  pose_out[6] = (d*h - e*g)/(a*e - b*d - a*f*h + b*f*g + c*d*h - c*e*g);
  pose_out[7] = -(a*h - b*g)/(a*e - b*d - a*f*h + b*f*g + c*d*h - c*e*g);
  T normalizer = (a*e - b*d)/(a*e - b*d - a*f*h + b*f*g + c*d*h - c*e*g);

  pose_out[0] /= normalizer;
  pose_out[1] /= normalizer;
  pose_out[2] /= normalizer;
  pose_out[3] /= normalizer;
  pose_out[4] /= normalizer;
  pose_out[5] /= normalizer;
  pose_out[6] /= normalizer;
  pose_out[7] /= normalizer;
}

template <typename T>
void applyPose8DOF (T x, T y, const T* const pose, T& x_, T& y_) {
  x_ = pose[0] * x + pose[1] * y + pose[2];
  y_ = pose[3] * x + pose[4] * y + pose[5];
  T normalizer = pose[6] * x + pose[7] * y + T(1.0f);
  x_ /= normalizer;
  y_ /= normalizer;
}



struct alignmentPair {
  alignmentPair(int n) {
    num_of_pts = n;
    kps_a = new float[n * 2];
    kps_b = new float[n * 2];
  }

  ~alignmentPair() {
    delete[] kps_a;
    delete[] kps_b;
  }

  int num_of_pts;
  int pose_a_id;
  int pose_b_id;
  float* kps_a;
  float* kps_b;
};

// interfaces
void appendAlignmentPairs(std::vector<int>& pose_id_a_array,
                          std::vector<int>& pose_id_b_array,
                          std::vector<Eigen::MatrixXf>& sift_loc_a_array,
                          std::vector<Eigen::MatrixXf>& sift_loc_b_array,
                          std::vector<alignmentPair*>& all_pairs);

void convertPoses(Eigen::MatrixXf& x_y_angle_array,
                  std::vector<double*>& all_poses);

void convertPosesBack(std::vector<double*>& all_poses,
                      Eigen::MatrixXf& x_y_angle_array);

void refinePoses (std::vector<alignmentPair*>& all_pairs,
                  std::vector<double*>& all_poses);


void refinePosesWithHardConstraints (std::vector<alignmentPair*>& all_pairs,
                                    std::vector<alignmentPair*>& all_fixed_constraint_pairs,
                                    std::vector<double*>& all_poses,
                                    std::vector<double*>& all_fixed_poses);

#endif
