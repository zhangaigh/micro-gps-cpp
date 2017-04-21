#include "image_func.h"
#include "flann/flann.h"
#include "util.h"

namespace MicroGPS {
namespace ImageFunc {

Image* cropPatch(const Image* im,
                      const float x, const float y, const float orientation,
                      const int win_width, const int win_height) {
  unsigned n_channels = im->channels();
  Image* patch = new Image (win_width, win_height, n_channels);
  uchar* patch_data = patch->data();
  
  float half_w = (float)win_width / 2;
  float half_h = (float)win_height / 2;

  // rotated x-axis
  float x_axis_x = cos(orientation);
  float x_axis_y = -sin(orientation);

  // rotated y-axis
  float y_axis_x = sin(orientation);
  float y_axis_y = cos(orientation);

  // [-half_w + 0.5, half_w - 0.5]
  unsigned pixel_cnt = 0;
  for (float patch_y = - half_h + 0.5; patch_y <= half_h - 0.5; patch_y += 1.0) {
    for (float patch_x = - half_w + 0.5; patch_x <= half_w - 0.5; patch_x += 1.0) {
      float sample_x = x + patch_x * x_axis_x + patch_y * y_axis_x;
      float sample_y = y + patch_x * x_axis_y + patch_y * y_axis_y;

      if (sample_x < 0 || sample_x >= im->width()-1 || sample_y < 0 || sample_y >= im->height()-1) {
        delete patch;
        printf("cropPatch(): not valid patch\n");
        return NULL;
      }

      for (unsigned ch = 0; ch < n_channels; ch++) {
        patch_data[pixel_cnt * n_channels + ch] = (uchar)(im->bilinearSample(sample_y, sample_x, ch));
      }
      pixel_cnt++;
    }
  }

  return patch; 
}

void matchFeatureBidirectional(Image* img1_ptr, Image* img2_ptr,
                              std::vector<int>& matched_idx1,
                              std::vector<int>& matched_idx2,
                              bool extract_feature) {

  if (extract_feature) {
    // TODO: let it work for other features
    img1_ptr->extractSIFT();
    img2_ptr->extractSIFT();
  }

  size_t num_img1_feat = img1_ptr->getNumLocalFeatures();
  size_t num_img2_feat = img2_ptr->getNumLocalFeatures();

  if (num_img1_feat == 0 || num_img2_feat == 0) {
    printf("matchFeatureBidirectional: num_img1_feat == 0 || num_img2_feat == 0");
  }

  size_t feat_dim = img1_ptr->getLocalFeature(0)->descriptor.size();

  // copy data and build index
  flann::Matrix<float> flann_feat1(new float[num_img1_feat * feat_dim],
                                  num_img1_feat, feat_dim);

  for (int idx = 0; idx < num_img1_feat; idx++) {
    LocalFeature* f = img1_ptr->getLocalFeature(idx);
    for (int d = 0; d < feat_dim; d++) {
      flann_feat1[idx][d] = f->descriptor(d);
    }
  }

  flann::Index<L2<float> > kdtree1(flann_feat1, KDTreeIndexParams());
  kdtree1.buildIndex();


  flann::Matrix<float> flann_feat2(new float[num_img2_feat * feat_dim],
                                  num_img2_feat, feat_dim);

  for (int idx = 0; idx < num_img2_feat; idx++) {
    LocalFeature* f = img2_ptr->getLocalFeature(idx);
    for (int d = 0; d < feat_dim; d++) {
      flann_feat2[idx][d] = f->descriptor(d);
    }
  }

  flann::Index<L2<float> > kdtree2(flann_feat2, KDTreeIndexParams());
  kdtree2.buildIndex();


  flann::Matrix<int> flann_index1(new int[2], 1, 2);
  flann::Matrix<float> flann_dist1(new float[2], 1, 2);

  flann::Matrix<int> flann_index2(new int[2], 1, 2);
  flann::Matrix<float> flann_dist2(new float[2], 1, 2);


  matched_idx1.resize(num_img1_feat);
  matched_idx2.resize(num_img1_feat);

  int cnt = 0;
  for (int idx1 = 0; idx1 < num_img1_feat; idx1++) {
    flann::Matrix<float> flann_query1(img1_ptr->getLocalFeature(idx1)->descriptor.data(), 1, feat_dim);
    kdtree2.knnSearch(flann_query1, flann_index1, flann_dist1, 2, flann::SearchParams(256));
    if (flann_dist1[0][0] < 0.6*0.6 * flann_dist1[0][1]) {
      int idx2 = flann_index1[0][0];
      flann::Matrix<float> flann_query2(img2_ptr->getLocalFeature(idx2)->descriptor.data(), 1, feat_dim);
      kdtree1.knnSearch(flann_query2, flann_index2, flann_dist2, 2, flann::SearchParams(256));
      if (flann_dist2[0][0] < 0.6*0.6 * flann_dist2[0][1] && flann_index2[0][0] == idx1) {
        matched_idx1[cnt] = idx1;
        matched_idx2[cnt] = idx2;

        // printf("%d <-> %d\n", idx1, idx2);
        cnt++;
      }
    }
  }
  matched_idx1.resize(cnt);
  matched_idx2.resize(cnt);

  printf("matched %d points\n", cnt);

  delete[] flann_feat1.ptr();
  delete[] flann_index1.ptr();
  delete[] flann_dist1.ptr();
  delete[] flann_feat2.ptr();
  delete[] flann_index2.ptr();
  delete[] flann_dist2.ptr();
}

// compute rigid transformation using matched keypoints
void computeRigidTransformation(Eigen::MatrixXf points1, Eigen::MatrixXf points2,
                                Eigen::MatrixXf& pose) {

  int n_points = points1.rows();
  int dimension = points1.cols();

  if (points2.rows() != n_points || points2.cols() != dimension) {
    printf("size of the points sets aren't equal\n");
    return;
  }

  Eigen::MatrixXf centroid1 = points1.colwise().sum() / (float)n_points;
  Eigen::MatrixXf centroid2 = points2.colwise().sum() / (float)n_points;

  for (int i = 0; i < n_points; i++) {
    points1.row(i) = points1.row(i) - centroid1;
    points2.row(i) = points2.row(i) - centroid2;
  }

  Eigen::MatrixXf S = points1.transpose() * points2;

  Eigen::JacobiSVD<Eigen::MatrixXf> svd(S, Eigen::ComputeThinV | Eigen::ComputeThinU);

  Eigen::MatrixXf M(dimension, dimension);
  M.setIdentity();

  M(dimension-1, dimension-1) = (svd.matrixV() * svd.matrixU().transpose()).determinant();

  Eigen::MatrixXf R = svd.matrixV() * M * svd.matrixU().transpose();
  Eigen::MatrixXf t = centroid2.transpose() - R * centroid1.transpose();

  pose = Eigen::MatrixXf(dimension+1, dimension+1);
  pose.setIdentity();
  pose.block(0, 0, dimension, dimension) = R;
  pose.block(0, dimension, dimension, 1) = t;
}

// RANSAC
void estimateRigidTransformationRANSAC(Eigen::MatrixXf points1, Eigen::MatrixXf points2,
                                        Eigen::MatrixXf& pose, std::vector<int>& inliers,
                                        int num_iterations,
                                        float error_thresh // 3 pixels
                                      ) {
  // test dimension
  int n_points = points1.rows();
  int dimension = points1.cols();
  if (points2.rows() != n_points || points2.cols() != dimension) {
    printf("size of the points sets aren't equal\n");
    return;
  }

  int min_num_points;
  if (dimension == 2) {
    min_num_points = 2;
  } else if (dimension == 3) {
    min_num_points = 3;
  }

  inliers.clear();

  for (int iter = 0; iter < num_iterations; iter++) {
    std::vector<int> sel(min_num_points);
    util::randomSample(n_points, min_num_points, sel);


    Eigen::MatrixXf points1_sel (min_num_points, dimension);
    Eigen::MatrixXf points2_sel (min_num_points, dimension);

    for (int point_idx = 0; point_idx < min_num_points; point_idx++) {
      points1_sel.row(point_idx) = points1.row(sel[point_idx]);
      points2_sel.row(point_idx) = points2.row(sel[point_idx]);
    }

    Eigen::MatrixXf test_pose;
    computeRigidTransformation(points1_sel, points2_sel, test_pose);


    int num_inliers = 0;
    std::vector<int> current_inliers;
    for (int point_idx = 0; point_idx < n_points; point_idx++) {
      Eigen::VectorXf dist = test_pose.block(0,0,dimension,dimension) * points1.row(point_idx).transpose() +
                            test_pose.block(0,dimension,dimension,1) - points2.row(point_idx);


      if (dist.norm() <= error_thresh) {
        current_inliers.push_back(point_idx);
      }
    }

    if (current_inliers.size() > inliers.size()) {
      pose = test_pose;
      inliers = current_inliers;
    }
  }

  Eigen::MatrixXf points1_sel (inliers.size(), dimension);
  Eigen::MatrixXf points2_sel (inliers.size(), dimension);
  for (int point_idx = 0; point_idx < inliers.size(); point_idx++) {
    points1_sel.row(point_idx) = points1.row(inliers[point_idx]);
    points2_sel.row(point_idx) = points2.row(inliers[point_idx]);
  }

  // finally estimate using all the inliers 
  computeRigidTransformation(points1_sel, points2_sel, pose);
}

// use matched features and estimate pose by RANSAC
bool estimatePoseFromMatchedImages(Image* img1_ptr, Image* img2_ptr,
                                    std::vector<int>& matched_idx1,
                                    std::vector<int>& matched_idx2,
                                    Eigen::MatrixXf& pose,
                                    std::vector<int>& inliers,
                                    int num_ransac_iterations,
                                    float error_thresh) { // 3 pixels

  int num_matches = matched_idx1.size();
  if (num_matches < 5) {
    printf("estimatePoseFromMatchedImages: too few matched features: %d...\n", num_matches);
    return false;
  }

  Eigen::MatrixXf points1 (num_matches, 2);
  Eigen::MatrixXf points2 (num_matches, 2);
  for (int i = 0; i < num_matches; i++) {
    LocalFeature* f1 = img1_ptr->getLocalFeature(matched_idx1[i]);
    LocalFeature* f2 = img2_ptr->getLocalFeature(matched_idx2[i]);
    points1(i, 0) = f1->x;
    points1(i, 1) = f1->y;
    points2(i, 0) = f2->x;
    points2(i, 1) = f2->y;
  }

  estimateRigidTransformationRANSAC(points2, points1,
                                    pose, inliers,
                                    num_ransac_iterations,
                                    error_thresh); // 3 pixels


  if (inliers.size() < 5) {
    printf("estimatePoseFromMatchedImages: too few inliers after RANSAC -> %ld inliers\n", inliers.size());
    return false;
  } else {
    printf("estimatePoseFromMatchedImages: RANSAC successful -> %ld inliers\n", inliers.size());
    return true;
  }
}







}
}