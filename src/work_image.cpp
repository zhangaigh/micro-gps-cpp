#ifdef _OPENMP
# include <omp.h>
#endif

#include "work_image.h"
#include "time.h"
#include "util.h"
// #define FLANN_USE_CUDA
#include "flann/flann.h"

#include "database.h"

// extern "C" {
//   #include "vl/sift.h"
// }


#ifdef USE_SIFT_GPU
#ifdef ON_MAC
#include "OpenGL/gl.h"
#else
#include "GL/gl.h"
#endif
// SiftGPU* g_sift_gpu = NULL;

SiftGPU g_sift_gpu;
void initSiftGPU() {
  // g_sift_gpu = new SiftGPU();
  char * sift_gpu_argv[] ={"-t", "0", "-v", "0", "-cuda"};
  g_sift_gpu.ParseParam(5, sift_gpu_argv); 
  int support = g_sift_gpu.CreateContextGL();
  if(support != SiftGPU::SIFTGPU_FULL_SUPPORTED) {
    return; 
  } else {
    printf("SiftGPU supported\n");
  }
}


#endif

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}


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

  // std::cout << "centroid1 = \n" << centroid1 << std::endl;
  // std::cout << "centroid2 = \n" << centroid2 << std::endl;


  for (int i = 0; i < n_points; i++) {
    points1.row(i) = points1.row(i) - centroid1;
    points2.row(i) = points2.row(i) - centroid2;
  }

  // std::cout << "points1 = \n" << points1 << std::endl;
  // std::cout << "points2 = \n" << points2 << std::endl;


  Eigen::MatrixXf S = points1.transpose() * points2;

  Eigen::JacobiSVD<Eigen::MatrixXf> svd(S, Eigen::ComputeThinV | Eigen::ComputeThinU);

  // std::cout << "Its singular values are:" << std::endl << svd.singularValues() << std::endl;
  // std::cout << "Its left singular vectors are the columns of the thin U matrix:" << std::endl << svd.matrixU() << std::endl;
  // std::cout << "Its right singular vectors are the columns of the thin V matrix:" << std::endl << svd.matrixV() << std::endl;

  // // Same results
  // Eigen::JacobiSVD<Eigen::MatrixXf> svd2(S, Eigen::ComputeFullV | Eigen::ComputeFullU);
  // std::cout << "Its singular values are:" << std::endl << svd2.singularValues() << std::endl;
  // std::cout << "Its left singular vectors are the columns of the thin U matrix:" << std::endl << svd2.matrixU() << std::endl;
  // std::cout << "Its right singular vectors are the columns of the thin V matrix:" << std::endl << svd2.matrixV() << std::endl;

  Eigen::MatrixXf M(dimension, dimension);
  M.setIdentity();

  // std::cout << "M = \n" << M << std::endl;
  M(dimension-1, dimension-1) = (svd.matrixV() * svd.matrixU().transpose()).determinant();

  Eigen::MatrixXf R = svd.matrixV() * M * svd.matrixU().transpose();
  Eigen::MatrixXf t = centroid2.transpose() - R * centroid1.transpose();


  // std::cout << "R = \n" << R << std::endl;
  // std::cout << "t = \n" << t << std::endl;

  pose = Eigen::MatrixXf(dimension+1, dimension+1);
  pose.setIdentity();
  pose.block(0, 0, dimension, dimension) = R;
  pose.block(0, dimension, dimension, 1) = t;

  // std::cout << "pose = \n" << pose << std::endl;

}

void estimateRigidTransformation(Eigen::MatrixXf points1, Eigen::MatrixXf points2,
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

  // Eigen::MatrixXf pose;
  // int best_num_inliers = 0;
  inliers.clear();

  for (int iter = 0; iter < num_iterations; iter++) {
    std::vector<int> sel(min_num_points);
    randomSample(n_points, min_num_points, sel);


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
        // num_inliers++;
        current_inliers.push_back(point_idx);
      }
    }
    // printf("num_inliers = %d\n", current_inliers.size());

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

  computeRigidTransformation(points1_sel, points2_sel, pose);


}

// TODO: precompute warp mapping, do not do actual warping
void computeWarpImageMapping(int im_width, int im_height,
                            int world_min_x, int world_min_y,
                            int world_max_x, int world_max_y,
                            Eigen::Matrix3f pose,
                            std::vector<Eigen::Vector2f>& world_location,
                            std::vector<Eigen::Vector2f>& im_location) {

  float lx = 0;
  float ux = (float)(im_width-1);
  float ly = 0;
  float uy = (float)(im_height-1);

  // transform four corners

  Eigen::MatrixXf corners(2, 4);
  corners(0, 0) = lx;
  corners(1, 0) = ly;
  corners(0, 1) = lx;
  corners(1, 1) = uy;
  corners(0, 2) = ux;
  corners(1, 2) = uy;
  corners(0, 3) = ux;
  corners(1, 3) = ly;

  corners = pose.block(0, 0, 2, 2) * corners;
  corners.row(0) = corners.row(0).array() + pose(0, 2);
  corners.row(1) = corners.row(1).array() + pose(1, 2);
  // corners = corners.colwise().add(pose.block(0, 2, 2, 1));

  Eigen::MatrixXf lowerbound = corners.rowwise().minCoeff();
  Eigen::MatrixXf upperbound = corners.rowwise().maxCoeff();


  // printf("boundaries: min_x = %f, min_y = %f, max_x = %f, max_y = %f\n",
  //         lowerbound(0), lowerbound(1), upperbound(0), upperbound(1));

  int min_x = std::max((int)floor(lowerbound(0)), world_min_x);
  int min_y = std::max((int)floor(lowerbound(1)), world_min_y);
  int max_x = std::min((int)floor(upperbound(0)), world_max_x);
  int max_y = std::min((int)floor(upperbound(1)), world_max_y);

  // printf("boundaries: min_x = %d, min_y = %d, max_x = %d, max_y = %d\n",
  //         min_x, min_y, max_x, max_y);

  world_location.clear();
  im_location.clear();
  int num_idx = (max_x - min_x + 1) * (max_y - min_y + 1);
  world_location.resize(num_idx);
  im_location.resize(num_idx);
  // printf("num_idx = %d\n", num_idx);

  Eigen::MatrixXf inv_pose = pose.inverse();

  int cnt = 0;
  for (int x = min_x; x <= max_x; x++) {
    for (int y = min_y; y <= max_y; y++) {
      // inverse mapping
      Eigen::MatrixXf pt(2, 1);
      pt(0, 0) = (float)x;
      pt(1, 0) = (float)y;
      pt = inv_pose.block(0, 0, 2, 2) * pt + inv_pose.block(0, 2, 2, 1);

      if (pt(0, 0) >= 0 && pt(0, 0) < im_width &&
          pt(1, 0) >= 0 && pt(1, 0) < im_height) {

        world_location[cnt](0) = (float)x;
        world_location[cnt](1) = (float)y;

        im_location[cnt](0) = pt(0, 0);
        im_location[cnt](1) = pt(1, 0);
        cnt++;
      }
    }
  }
  world_location.resize(cnt);
  im_location.resize(cnt);

  printf("computeWarpImageMapping(): computed %d mapping\n", cnt);
}


void warpImage(WorkImage* image,
              int world_min_x, int world_min_y,
              int world_max_x, int world_max_y,
              Eigen::Matrix3f pose) {

  std::vector<Eigen::Vector2f> world_location;
  std::vector<Eigen::Vector2f> im_location;

  int im_width = image->width();
  int im_height = image->height();

  computeWarpImageMapping(im_width, im_height,
                          world_min_x, world_min_y,
                          world_max_x, world_max_y,
                          pose,
                          world_location,
                          im_location);

  int world_width = world_max_x - world_min_x + 1;
  int world_height = world_max_y - world_min_y + 1;

  WorkImage* world_image = new WorkImage(world_width, world_height);

  uchar* world_data = world_image->data();
  uchar* im_data = image->data();

  for (int i = 0; i < world_location.size(); i++) {
    // TODO bilinear sampling
    int world_x = world_location[i](0) - world_min_x;
    int world_y = world_location[i](1) - world_min_y;
    int im_x = round(im_location[i](0));
    int im_y = round(im_location[i](1));

    // printf("world_x = %d, world_y = %d, im_x = %d, im_y = %d\n",
    //       world_x, world_y, im_x, im_y);

    for (int c = 0; c < 3; c++) {
      world_data[(world_y * world_width + world_x) * 3 + c] =
        im_data[(im_y * im_width + im_x) * 3 + c];
    }
  }

  world_image->write("warped_image.png");

  delete world_image;
}

void computeImageArrayWorldSize(std::vector<WorkImage*>& images,
                                std::vector<Eigen::Matrix3f>& poses,
                                float warp_scale,
                                int& world_min_x,
                                int& world_min_y,
                                int& world_max_x,
                                int& world_max_y) {
  int n_images = poses.size();

  images[0]->loadImage();
  images[0]->resize(warp_scale);
  int im_width = images[0]->width();
  int im_height = images[0]->height();
  images[0]->release();

  float lx = 0;
  float ux = (float)(im_width-1);
  float ly = 0;
  float uy = (float)(im_height-1);

  Eigen::MatrixXf all_corners(2, n_images * 4);
  for (int i = 0; i < n_images; i++) {
    // std::cout << poses[i] << std::endl;

    // resize pose as well
    poses[i].block(0, 2, 2, 1) *= warp_scale;

    // transform four corners
    Eigen::MatrixXf corners(2, 4);
    corners(0, 0) = lx;
    corners(1, 0) = ly;
    corners(0, 1) = lx;
    corners(1, 1) = uy;
    corners(0, 2) = ux;
    corners(1, 2) = uy;
    corners(0, 3) = ux;
    corners(1, 3) = ly;

    // std::cout << corners << std::endl;

    corners = poses[i].block(0, 0, 2, 2) * corners;
    corners.row(0) = corners.row(0).array() + poses[i](0, 2);
    corners.row(1) = corners.row(1).array() + poses[i](1, 2);

    // std::cout << corners << std::endl;

    all_corners.block(0, i * 4, 2, 4) = corners;
  }
  // std::cout << all_corners << std::endl;


  Eigen::MatrixXf lowerbound = all_corners.rowwise().minCoeff();
  Eigen::MatrixXf upperbound = all_corners.rowwise().maxCoeff();

  world_min_x = floor(lowerbound(0));
  world_min_y = floor(lowerbound(1));
  world_max_x = ceil(upperbound(0));
  world_max_y = ceil(upperbound(1));
}


WorkImage* warpImageArray(std::vector<WorkImage*>& images,
                    std::vector<Eigen::Matrix3f>& poses,
                    float warp_scale) {

  int n_images = poses.size();

  if (n_images <= 0) {
    printf("warpImageArray(): images array empty!\n");
  } else {
    printf("warpImageArray(): input %d images\n", n_images);
  }


  int world_min_x;
  int world_min_y;
  int world_max_x;
  int world_max_y;

  computeImageArrayWorldSize(images, poses, warp_scale,
                              world_min_x, world_min_y,
                              world_max_x, world_max_y);

  images[0]->loadImage();
  images[0]->resize(warp_scale);
  int im_width = images[0]->width();
  int im_height = images[0]->height();
  images[0]->release();
  //
  // float lx = 0;
  // float ux = (float)(im_width-1);
  // float ly = 0;
  // float uy = (float)(im_height-1);
  //
  // Eigen::MatrixXf all_corners(2, n_images * 4);
  // for (int i = 0; i < n_images; i++) {
  //   // std::cout << poses[i] << std::endl;
  //
  //   // resize pose as well
  //   poses[i].block(0, 2, 2, 1) *= warp_scale;
  //
  //   // transform four corners
  //   Eigen::MatrixXf corners(2, 4);
  //   corners(0, 0) = lx;
  //   corners(1, 0) = ly;
  //   corners(0, 1) = lx;
  //   corners(1, 1) = uy;
  //   corners(0, 2) = ux;
  //   corners(1, 2) = uy;
  //   corners(0, 3) = ux;
  //   corners(1, 3) = ly;
  //
  //   // std::cout << corners << std::endl;
  //
  //   corners = poses[i].block(0, 0, 2, 2) * corners;
  //   corners.row(0) = corners.row(0).array() + poses[i](0, 2);
  //   corners.row(1) = corners.row(1).array() + poses[i](1, 2);
  //
  //   // std::cout << corners << std::endl;
  //
  //   all_corners.block(0, i * 4, 2, 4) = corners;
  // }
  // // std::cout << all_corners << std::endl;
  //
  //
  // Eigen::MatrixXf lowerbound = all_corners.rowwise().minCoeff();
  // Eigen::MatrixXf upperbound = all_corners.rowwise().maxCoeff();
  //
  // int world_min_x = floor(lowerbound(0));
  // int world_min_y = floor(lowerbound(1));
  // int world_max_x = ceil(upperbound(0));
  // int world_max_y = ceil(upperbound(1));

  // printf("warpImageArray: world_min_x = %d, world_min_y = %d, world_max_x = %d, world_max_y = %d\n",
  //         world_min_x, world_min_y, world_max_x, world_max_y);

  int world_width = world_max_x - world_min_x + 1;
  int world_height = world_max_y - world_min_y + 1;

  // int x_step = world_width / n_x_blocks;
  // int y_step = world_height / n_y_blocks;

  int* world_pixel_weight = new int[world_width * world_height];
  int* world_data_sum = new int[world_width * world_height * 3];
  for (int pixel_idx = 0; pixel_idx < world_width*world_height; pixel_idx++) {
    world_pixel_weight[pixel_idx] = 0;
    world_data_sum[pixel_idx * 3 + 0] = 0;
    world_data_sum[pixel_idx * 3 + 1] = 0;
    world_data_sum[pixel_idx * 3 + 2] = 0;
  }

  for (int im_idx = 0; im_idx < n_images; im_idx++) {
    printf("warpImageArray(): warping image %d\n", im_idx);

    std::vector<Eigen::Vector2f> world_location;
    std::vector<Eigen::Vector2f> im_location;

    computeWarpImageMapping(im_width, im_height,
                            world_min_x, world_min_y,
                            world_max_x, world_max_y,
                            poses[im_idx],
                            world_location,
                            im_location);


    images[im_idx]->loadImage();
    images[im_idx]->resize(warp_scale);

    uchar* im_data = images[im_idx]->data();

    for (int loc_idx = 0; loc_idx < world_location.size(); loc_idx++) {
      int world_x = world_location[loc_idx](0) - world_min_x;
      int world_y = world_location[loc_idx](1) - world_min_y;
      int im_x = round(im_location[loc_idx](0));
      int im_y = round(im_location[loc_idx](1));

      if (im_x >= 0 && im_y >= 0 && im_x < im_width && im_y < im_height) {
        world_pixel_weight[world_y * world_width + world_x] += 1;

        for (int c = 0; c < 3; c++) {
          world_data_sum[(world_y * world_width + world_x) * 3 + c] +=
                        (int)(im_data[(im_y * im_width + im_x) * 3 + c]);
        }
      }

    }
    images[im_idx]->release();
  }


  WorkImage* world_image = new WorkImage(world_width, world_height);
  uchar* world_data = world_image->data();

  for (int pixel_idx = 0; pixel_idx < world_width*world_height; pixel_idx++) {
    for (int c = 0; c < 3; c++) {
      world_data[pixel_idx * 3 + c] = (uchar)((double)world_data_sum[pixel_idx * 3 + c] / (double)(world_pixel_weight[pixel_idx]));
    }
  }

  WorkImage* world_image_weight = new WorkImage(world_width, world_height, false);
  uchar* world_image_weight_data = world_image_weight->data();

  for (int pixel_idx = 0; pixel_idx < world_width*world_height; pixel_idx++) {
    world_image_weight_data[pixel_idx] = (uchar)world_pixel_weight[pixel_idx];
  }

  delete[] world_data_sum;
  delete[] world_pixel_weight;

  // if (output_path) {
  //   world_image->write(output_path);
  // }
  // world_image_weight->write("world_weight.png");
  // delete world_image;

  delete world_image_weight;

  return world_image;
}








WorkImage::WorkImage(const char* image_path, const char* precomputed_feature_path)
{
  sprintf(m_image_path, "%s", image_path);
  printf("WorkImage::WorkImage(): image_path = %s\n", image_path);
  m_data = NULL;
  m_sift_detector = NULL;
  m_is_color = true;

  if (precomputed_feature_path != NULL) {
    sprintf(m_external_feature_path, "%s", precomputed_feature_path);
    printf("precomputed feature path = %s\n", precomputed_feature_path);
  } else {
    sprintf(m_external_feature_path, "");  
  }
}

WorkImage::WorkImage(int width, int height, bool is_color):
m_width(width),
m_height(height)
{
  if (is_color) {
    m_data = new uchar[width * height * 3];
  } else {
    m_data = new uchar[width * height];
  }
  m_is_color = is_color;
  m_sift_detector = NULL;
  // g_sift_gpu = NULL;
  sprintf(m_image_path, "");  
  sprintf(m_external_feature_path, "");
}

WorkImage::~WorkImage() {
  if (m_sift_detector) {
    delete m_sift_detector;
    m_sift_detector = NULL;
  }

  // if (g_sift_gpu) {
  //   delete g_sift_gpu;
  //   g_sift_gpu = NULL;
  // }

  if (m_data) {
    delete[] m_data;
    m_data = NULL;
  }

  for (int i = 0; i < m_sift_features.size(); i++) {
    delete m_sift_features[i];
  }
  m_sift_features.clear();
}

void WorkImage::release() {
  // release memory consuming members
  if (m_data) {
    // printf("WorkImage::release(): releasing m_data\n");
    delete[] m_data;
    m_data = NULL;
  }

  // release SIFT features as well
  // printf("WorkImage::release(): releasing m_sift_features\n");
  for (int i = 0; i < m_sift_features.size(); i++) {
    delete m_sift_features[i];
  }
  m_sift_features.clear();
}


void WorkImage::loadImage() {
  printf("WorkImage::loadImage(): loading %s\n", m_image_path);
  cv::Mat bgr = cv::imread(m_image_path, CV_LOAD_IMAGE_COLOR);
  printf("WorkImage::loadImage(): image loaded: %d x %d x %d\n", bgr.rows, bgr.cols, bgr.channels());

  m_width = bgr.cols;
  m_height = bgr.rows;

  // reset m_data anyway
  if (m_data) {
    delete[] m_data;
    m_data = NULL;
  }
  m_data = new uchar[m_width * m_height * 3];
  memcpy(m_data, bgr.data, m_width * m_height * 3);
}


void WorkImage::toGray() {
  cv::Mat bgr(m_height, m_width, CV_8UC3, m_data);

  cv::Mat gray;
  cv::cvtColor(bgr, gray, CV_BGR2GRAY);

  uchar* m_data_old = m_data;
  m_data = new uchar[m_width * m_height];
  memcpy(m_data, gray.data, m_width * m_height);
  delete[] m_data_old; // clear old buffer
  m_data_old = NULL;

}

void WorkImage::resize(float ratio) {
  cv::Mat bgr;
  if (m_is_color) {
    bgr = cv::Mat(m_height, m_width, CV_8UC3, m_data);
  } else {
    bgr = cv::Mat(m_height, m_width, CV_8UC1, m_data);
  }
  cv::Mat bgr_small;

  cv::resize(bgr, bgr_small, cv::Size(), ratio, ratio);

  printf("WorkImage::resize(): resized to %d x %d x %d\n", bgr_small.rows, bgr_small.cols, bgr_small.channels());

  m_width = bgr_small.cols;
  m_height = bgr_small.rows;

  uchar* m_data_old = m_data;
  if (m_is_color) {
    m_data = new uchar[m_width * m_height * 3];
    memcpy(m_data, bgr_small.data, m_width * m_height * 3);
  } else {
    m_data = new uchar[m_width * m_height];
    memcpy(m_data, bgr_small.data, m_width * m_height );
  }
  delete[] m_data_old; // clear old buffer
  m_data_old = NULL;
}

void WorkImage::write(const char* image_path) {
  cv::Mat cv_im;
  if (m_is_color) {
    cv_im = cv::Mat(m_height, m_width, CV_8UC3, m_data);
  } else {
    cv_im = cv::Mat(m_height, m_width, CV_8UC1, m_data);
  }

  printf("WorkImage::write(): writing image to %s\n", image_path);
  cv::imwrite(image_path, cv_im);
}

bool WorkImage::loadPrecomputedFeature(float downsampling) {
  if (strcmp(m_external_feature_path, "") == 0) {
    printf("cannot read external feature!\n");
    return false;
  }
  printf("WorkImage::loadPrecomputedFeature(): loading external features %s\n", m_external_feature_path);
  
  FILE* fp = fopen(m_external_feature_path, "r");
  int size[2];
  fread(size, sizeof(int), 2, fp);
  int chunk_size = size[1] + 4; // 4 floats for locations
  float* buffer = new float[size[0] * chunk_size];  
  fread(buffer, sizeof(float), chunk_size * size[0], fp);
  fclose(fp);
  printf("size[] = %d x %d\n", size[0], size[1]);

  // assign values
  m_sift_features.resize(size[0]);
  for (int i = 0; i < size[0]; i++) {
    SIFTFeature* f = new SIFTFeature;
    f->x = buffer[i * chunk_size + 0] / downsampling;
    f->y = buffer[i * chunk_size + 1] / downsampling;
    f->scale = buffer[i * chunk_size + 2] / downsampling;
    f->angle = buffer[i * chunk_size + 3];
    if (f->scale < 1.0) {
      printf("%d: scale = %f!\n", i, f->scale);
      exit(-1);
    }
    f->descriptor = Eigen::Map<Eigen::RowVectorXf>(buffer + chunk_size * i + 4, size[1]);

    // printf("i = %d\n", i);
    f->local_pose(0, 0) = cos(-f->angle);
    f->local_pose(1, 0) = sin(-f->angle);
    f->local_pose(2, 0) = 0;
    f->local_pose(0, 1) = -sin(-f->angle);
    f->local_pose(1, 1) = cos(-f->angle);
    f->local_pose(2, 1) = 0;
    f->local_pose(0, 2) = f->x;
    f->local_pose(1, 2) = f->y;
    f->local_pose(2, 2) = 1;

    m_sift_features[i] = f;
  }
  delete[] buffer;  

  return true;
}


#ifdef USE_SIFT_GPU
void WorkImage::extractSIFT(float downsampling) {
  printf("USING SIFTGPU\n");

  cv::Mat gray_small;

  if (m_is_color) {
    cv::Mat bgr(m_height, m_width, CV_8UC3, m_data);
    cv::cvtColor(bgr, gray_small, CV_BGR2GRAY);
    cv::resize(gray_small, gray_small, cv::Size(), downsampling, downsampling);
  } else {
    cv::Mat gray(m_height, m_width, CV_8UC1, m_data);
    cv::resize(gray, gray_small, cv::Size(), downsampling, downsampling);  
  }

  printf("start running sift gpu\n");
  g_sift_gpu.RunSIFT (gray_small.cols, gray_small.rows, gray_small.data, GL_LUMINANCE, GL_UNSIGNED_BYTE); 
  
  printf("finish running sift\n");
  int num = g_sift_gpu.GetFeatureNum();//get feature count
  printf("detected %d features\n", num);

  // std::vector<float> descriptors(128*num);
  float* descriptors = new float[128 * num]; 
  std::vector<SiftGPU::SiftKeypoint> keypoints(num);
  //read back keypoints and normalized descritpros
  //specify NULL if you don’t need keypionts or descriptors
  g_sift_gpu.GetFeatureVector(&keypoints[0], &descriptors[0]);


  for (int i = 0; i < m_sift_features.size(); i++) {
    delete m_sift_features[i];  
  }
  m_sift_features.resize(num);


  for (int i = 0; i < num; i++) {
    // printf("reading feature %d\n", i);
    SIFTFeature* f = new SIFTFeature;
    f->x = keypoints[i].x / downsampling;
    f->y = keypoints[i].y / downsampling;
    f->angle = keypoints[i].o;
    f->scale = keypoints[i].s / downsampling;
    f->strength = 0.0f;
    f->descriptor = Eigen::Map<Eigen::RowVectorXf>((float*)(descriptors + 128 * i), 128);


    // double angle = keypoints[i].o / 180 * M_PI; //convert to radian
    double angle = -keypoints[i].o; //siftgpu uses different conventions, we need to flip the sign of the angle
    // printf("angle = %f\n", angle);

    // compute local pose
    f->local_pose(0, 0) = cos(angle);
    f->local_pose(1, 0) = sin(angle);
    f->local_pose(2, 0) = 0;
    f->local_pose(0, 1) = -sin(angle);
    f->local_pose(1, 1) = cos(angle);
    f->local_pose(2, 1) = 0;
    f->local_pose(0, 2) = f->x;
    f->local_pose(1, 2) = f->y;
    f->local_pose(2, 2) = 1;

    m_sift_features[i] = f;
  }
  delete[] descriptors;

  printf("WorkImage::extractSIFT(): detected %lu keypoints\n", m_sift_features.size());

}

#endif


void WorkImage::saveSIFTFeatures(char* path, int num_samples, float margin_thresh) {
  FILE* fp = fopen(path, "w");

  std::vector<SIFTFeature*> valid_features;
  int num_features = m_sift_features.size();
  for (int i = 0; i < num_features; i++) {
    SIFTFeature* f = m_sift_features[i];
    if (f->x > margin_thresh && f->x < m_width-1-margin_thresh &&
        f->y > margin_thresh && f->y < m_height-1-margin_thresh) {
      valid_features.push_back(f);
    }
  }

  num_features = valid_features.size();
  std::vector<int> sel;
  randomSample(num_features, num_samples, sel);
  num_features = sel.size();

  fwrite(&num_features, sizeof(int), 1, fp);

  for (int i = 0; i < num_features; i++) {
    SIFTFeature* f = valid_features[sel[i]];
    float kp[4];
    kp[0] = f->x;
    kp[1] = f->y;
    kp[2] = f->scale;
    kp[3] = f->angle;
    fwrite(kp, sizeof(float), 4, fp);   
    fwrite(f->descriptor.data(), sizeof(float), f->descriptor.size(), fp);    
  }

  fclose(fp);
}



void WorkImage::PCADimReduction(Eigen::MatrixXf& PCA_basis) {
  for (int i = 0; i < m_sift_features.size(); i++) {
    SIFTFeature* f = m_sift_features[i];
    f->descriptor_PCAed = f->descriptor * PCA_basis; // still row vector
  }
  printf("WorkImage::PCADimReduction(); dimension reduced for work image\n");
}


void WorkImage::siftMatch(WorkImage* img,
                          std::vector<int>& matched_idx1,
                          std::vector<int>& matched_idx2) {

  WorkImage* img1_ptr = this;
  WorkImage* img2_ptr = img;
  img1_ptr->extractSIFT();
  img2_ptr->extractSIFT();

  // copy data and build index
  flann::Matrix<float> flann_feat1(new float[img1_ptr->getSIFTSize() * 128],
                                  img1_ptr->getSIFTSize(), 128);

  for (int idx = 0; idx < img1_ptr->getSIFTSize(); idx++) {
    SIFTFeature* f = img1_ptr->getSIFTFeature(idx);
    for (int d = 0; d < 128; d++) {
      flann_feat1[idx][d] = f->descriptor(d);
    }
  }

  flann::Index<L2<float> > kdtree1(flann_feat1, KDTreeIndexParams());
  kdtree1.buildIndex();


  flann::Matrix<float> flann_feat2(new float[img2_ptr->getSIFTSize() * 128],
                                  img2_ptr->getSIFTSize(), 128);

  for (int idx = 0; idx < img2_ptr->getSIFTSize(); idx++) {
    SIFTFeature* f = img2_ptr->getSIFTFeature(idx);
    for (int d = 0; d < 128; d++) {
      flann_feat2[idx][d] = f->descriptor(d);
    }
  }

  flann::Index<L2<float> > kdtree2(flann_feat2, KDTreeIndexParams());
  kdtree2.buildIndex();


  flann::Matrix<int> flann_index1(new int[2], 1, 2);
  flann::Matrix<float> flann_dist1(new float[2], 1, 2);

  flann::Matrix<int> flann_index2(new int[2], 1, 2);
  flann::Matrix<float> flann_dist2(new float[2], 1, 2);


  // std::vector<int> matched_idx1 (img1_ptr->getSIFTSize());
  // std::vector<int> matched_idx2 (img1_ptr->getSIFTSize());

  matched_idx1.resize(img1_ptr->getSIFTSize());
  matched_idx2.resize(img1_ptr->getSIFTSize());

  int cnt = 0;
  for (int idx1 = 0; idx1 < img1_ptr->getSIFTSize(); idx1++) {
    flann::Matrix<float> flann_query1(img1_ptr->getSIFTFeature(idx1)->descriptor.data(), 1, 128);
    kdtree2.knnSearch(flann_query1, flann_index1, flann_dist1, 2, flann::SearchParams(256));
    if (flann_dist1[0][0] < 0.6*0.6 * flann_dist1[0][1]) {
      int idx2 = flann_index1[0][0];
      flann::Matrix<float> flann_query2(img2_ptr->getSIFTFeature(idx2)->descriptor.data(), 1, 128);
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


bool WorkImage::siftMatchEstimatePose(WorkImage* img,
                                      std::vector<int>& matched_idx1,
                                      std::vector<int>& matched_idx2,
                                      Eigen::MatrixXf& pose,
                                      std::vector<int>& inliers) {
  WorkImage* img1_ptr = this;
  WorkImage* img2_ptr = img;

  int num_matches = matched_idx1.size();

  // if (num_matches < 5) {
  //   printf("WorkImage::siftMatchEstimatePose: too few matches: %d\n", num_matches);
  //   return false;
  // }

  Eigen::MatrixXf points1 (num_matches, 2);
  Eigen::MatrixXf points2 (num_matches, 2);
  for (int i = 0; i < num_matches; i++) {
    SIFTFeature* f1 = img1_ptr->getSIFTFeature(matched_idx1[i]);
    SIFTFeature* f2 = img2_ptr->getSIFTFeature(matched_idx2[i]);
    points1(i, 0) = f1->x;
    points1(i, 1) = f1->y;
    points2(i, 0) = f2->x;
    points2(i, 1) = f2->y;
  }


  // Eigen::MatrixXf pose;
  // std::vector<int> inliers;
  estimateRigidTransformation(points2, points1,
                              pose, inliers,
                              1000,
                              3.0f); // 3 pixels


  if (inliers.size() < 5) {
    printf("WorkImage::siftMatchEstimatePose: too few inliers after RANSAC -> %ld inliers\n", inliers.size());
    return false;
  } else {
    printf("WorkImage::siftMatchEstimatePose: RANSAC successful -> %ld inliers\n", inliers.size());
    return true;
  }
}




// test scripts (not maintained)


int test_single_image(int argc, char const *argv[]) {
  std::string p = "/Users/lgzhang/Documents/DATA/micro_gps_packed/fc_hallway_long_packed/test/frame000001.jpg";
  WorkImage* work_image = new WorkImage(p.c_str());

  clock_t begin, end;

  begin = clock();
  work_image->loadImage();
  end = clock();
  printf("elapsed time for loading image: %f secs\n", double(end - begin) / CLOCKS_PER_SEC);

  begin = clock();
  work_image->extractSIFT();
  end = clock();
  printf("elapsed time for SIFT: %f secs\n", double(end - begin) / CLOCKS_PER_SEC);



  return 0;
}


int test_rigid(int argc, char const *argv[]) {
  int n_inliers = 80;
  int n_outliers = 20;

  Eigen::MatrixXf points1 = Eigen::MatrixXf::Random(n_inliers,3);
  // Eigen::MatrixXf points2 = Eigen::MatrixXf::Random(5,3);
  // Eigen::MatrixXf points2 = points1 + Eigen::MatrixXf::Random(n_inliers,3) / 10000.0f;
  // Eigen::MatrixXf points2 = points1;


  // random rotation basis
  Eigen::Vector3f x_axis = Eigen::MatrixXf::Random(3, 1);
  x_axis.normalize();
  Eigen::Vector3f y_axis = Eigen::MatrixXf::Random(3, 1);
  y_axis = y_axis - x_axis * (y_axis.dot(x_axis));
  y_axis.normalize();
  Eigen::Vector3f z_axis = x_axis.cross(y_axis);
  Eigen::MatrixXf translation = Eigen::MatrixXf::Random(3, 1);

  Eigen::MatrixXf Rt(4, 4);
  Rt.setIdentity();
  Rt.block(0, 0, 3, 1) = x_axis;
  Rt.block(0, 1, 3, 1) = y_axis;
  Rt.block(0, 2, 3, 1) = z_axis;
  Rt.block(0, 3, 3, 1) = translation;


  std::cout << "fake Rt = \n" << Rt << std::endl;


  std::cout << "points1 = \n" << points1 << std::endl;
  Eigen::MatrixXf points2 = Rt.block(0,0,3,3) * points1.transpose();
  for (int i = 0; i < points2.cols(); i++) {
    points2.col(i) += Rt.block(0,3,3,1);
  }
  points2.transposeInPlace();
  std::cout << "points2 = \n" << points2 << std::endl;


  Eigen::MatrixXf points1_with_outliers = Eigen::MatrixXf::Random(n_inliers+n_outliers,3);
  Eigen::MatrixXf points2_with_outliers = Eigen::MatrixXf::Random(n_inliers+n_outliers,3);

  std::vector<int> inliers_idx;
  randomSample(n_inliers+n_outliers, n_inliers, inliers_idx);

  for (size_t i = 0; i < inliers_idx.size(); i++) {
    points1_with_outliers.row(inliers_idx[i]) = points1.row(i);
    points2_with_outliers.row(inliers_idx[i]) = points2.row(i);
  }



  Eigen::MatrixXf pose;
  // computeRigidTransformation(points1, points2, pose);

  std::vector<int> estimated_inliers;
  // estimateRigidTransformation(points1, points2, pose, estimated_inliers);
  estimateRigidTransformation(points1_with_outliers, points2_with_outliers, pose, estimated_inliers, 1000, 0.01);
  std::cout << "pose = \n" << pose << std::endl;
  std::cout << "# estimated_inliers = " << estimated_inliers.size() << std::endl;

  std::sort(inliers_idx.begin(), inliers_idx.end());
  printf("GT inliers = \n");
  for (size_t i = 0; i < inliers_idx.size(); i++) {
    printf("%d ", inliers_idx[i]);
  }
  printf("\n");


  printf("estimated inliers = \n");
  for (size_t i = 0; i < estimated_inliers.size(); i++) {
    printf("%d ", estimated_inliers[i]);
  }
  printf("\n");

  return 0;
}

int test_randsample(int argc, char const *argv[]) {
  for (int i = 0; i < 10; i++) {
    std::vector<int> sel(3);
    randomSample(10, 3, sel);
    printf("%d, %d, %d\n", sel[0], sel[1], sel[2]);
  }


  return 0;
}

Eigen::Matrix3f generate_rigid(float angle, float x, float y) {
  angle = angle / 180.0f * M_PI;
  Eigen::Matrix3f pose;
  pose(0, 0) = cos(angle);
  pose(1, 0) = sin(angle);
  pose(2, 0) = 0;
  pose(0, 1) = -sin(angle);
  pose(1, 1) = cos(angle);
  pose(2, 0) = 0;
  pose(0, 2) = x;
  pose(1, 2) = y;
  pose(2, 2) = 1;
  return pose;
}

int test_warp_image(int argc, char const *argv[]) {
  Eigen::Matrix3f pose1 = generate_rigid(45.0f, 0.0f, 0.0f);

  std::string p = "test_frame.jpg";
  WorkImage* work_image = new WorkImage(p.c_str());
  work_image->loadImage();


  // warpImage(work_image,
  //           -2000.0f, -2000.0f,
  //           2000.0f, 2000.0f,
  //           pose1);

  warpImage(work_image,
            -200.0f, -200.0f,
            200.0f, 200.0f,
            pose1);


  delete work_image;

  return 0;
}


int test_warp_image_array(int argc, char const *argv[]) {
  std::string s = "/Users/lgzhang/Documents/DATA/micro_gps_packed/fc_hallway_long_packed";
  Database* database = new Database(s.c_str());
  database->loadDatabase();
  database->loadDefaultTestSequence();

  int n_images = database->getDatabaseSize();
  // int n_images = 10;

  std::vector<WorkImage*> work_images(n_images);
  std::vector<Eigen::Matrix3f> work_image_poses(n_images);

  for (int i = 0; i < n_images; i++) {
    work_images[i] = new WorkImage(database->getDatabaseImage(i));
    work_image_poses[i] = database->getDatabasePose(i);
  }

  printf("all work images loaded\n");

  warpImageArray(work_images, work_image_poses, 1.0)->write("world.png");

  return 0;
}

int test_sift_matching(int argc, char const *argv[]) {

  WorkImage* work_image1 = new WorkImage("frame_a.jpg");
  WorkImage* work_image2 = new WorkImage("frame_b.jpg");
  work_image1->loadImage();
  work_image2->loadImage();

  std::vector<int> matched_idx1;
  std::vector<int> matched_idx2;
  work_image1->siftMatch(work_image2, matched_idx1, matched_idx2);


  Eigen::MatrixXf pose;
  std::vector<int> inliers;
  work_image1->siftMatchEstimatePose(work_image2, matched_idx1, matched_idx2,
                                    pose, inliers);

  std::vector<WorkImage*> work_images(2);
  std::vector<Eigen::Matrix3f> work_image_poses(2);

  work_images[0] = work_image1;
  work_images[1] = work_image2;
  work_image_poses[0].setIdentity();
  work_image_poses[1] = pose;
  printf("start warping\n");

  warpImageArray(work_images, work_image_poses, 1.0)->write("world.png");

  return 0;
}

int test_load_image(int argc, char const *argv[]) {
  WorkImage* work_image = new WorkImage("frame_a.jpg");

  for (int i = 0; i < 100; i++) {
    work_image->loadImage();
    work_image->extractSIFT();
    work_image->resize(0.5);
    work_image->release();
  }

  delete work_image;
  return 0;

}

int test_openmp(int argc, char const *argv[]) {
  #ifdef _OPENMP
  printf("openmp defined\n");
  #endif
  #pragma omp parallel num_threads(3)
  {
    #pragma omp parallel for
    for (int i = 0; i < 10; i++) {
      printf("%d\n", i);
    }
  }
  return 0;
}

// int main(int argc, char const *argv[]) {
//   std::srand(unsigned(std::time(0)));
//
//
//   // test_rigid(argc, argv);
//   // test_randsample(argc, argv);
//   // test_warp_image_array(argc, argv);
//   // test_openmp(argc, argv);
//   // test_sift_matching(argc, argv);
//   test_load_image(argc, argv);
//
//
//   return 0;
// }
