#include "image_func.h"


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
  float x_axis_y = sin(orientation);

  // rotated y-axis
  float y_axis_x = -sin(orientation);
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

        // printf("%d <-> %d: dist = %f\n", idx1, idx2, sqrt(flann_dist2[0][0]));
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

// compute the world limits for an image array
void computeImageArrayWorldLimits(std::vector<Image*>& images,
                                  std::vector<Eigen::Matrix3f>& poses,
                                  float warp_scale,
                                  int& world_min_x,
                                  int& world_min_y,
                                  int& world_max_x,
                                  int& world_max_y,
                                  bool dynamic_size) {
  int n_images = (int)poses.size();

  int im_width;
  int im_height;
  
  if (!dynamic_size) { // load image once to get dimension
    images[0]->loadImage();
    images[0]->resize(warp_scale);
    im_width = images[0]->width();
    im_height = images[0]->height();
    images[0]->release();
  }

  float lx = 0;
  float ux;
  float ly = 0;
  float uy;

  Eigen::MatrixXf all_corners(2, n_images * 4);
  for (int i = 0; i < n_images; i++) {
    // std::cout << poses[i] << std::endl;

    // resize pose as well
    poses[i].block(0, 2, 2, 1) *= warp_scale;

    if (dynamic_size) {
      images[i]->loadImage();
      images[i]->resize(warp_scale);
      im_width = images[i]->width();
      im_height = images[i]->height();
      images[i]->release();
    }

    ux = (float)(im_width-1);
    uy = (float)(im_height-1);

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

// precompute warp mapping, do not do actual warping
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

  // printf("computeWarpImageMapping(): computed %d mapping\n", cnt);
}

// warp an image array to generate a map
Image* warpImageArray(std::vector<Image*>& images,
                      std::vector<Eigen::Matrix3f>& poses,
                      float warp_scale,
                      bool dynamic_size) {
  
  int n_images = (int)poses.size();

  if (n_images <= 0) {
    printf("warpImageArray(): images array empty!\n");
  } else {
    printf("warpImageArray(): input %d images\n", n_images);
  }


  int world_min_x;
  int world_min_y;
  int world_max_x;
  int world_max_y;

  computeImageArrayWorldLimits(images, poses, warp_scale,
                               world_min_x, world_min_y,
                               world_max_x, world_max_y,
                               dynamic_size);

  size_t im_width;    
  size_t im_height;
  size_t im_channels;






  images[0]->loadImage();
  images[0]->resize(warp_scale);
  im_width    = images[0]->width();
  im_height   = images[0]->height();
  im_channels = images[0]->channels();
  images[0]->release();


  int world_width = world_max_x - world_min_x + 1;
  int world_height = world_max_y - world_min_y + 1;

  int* world_pixel_weight = new int[world_width * world_height];
  int* world_data_sum = new int[world_width * world_height * im_channels];
  for (size_t pixel_idx = 0; pixel_idx < world_width*world_height; pixel_idx++) {
    world_pixel_weight[pixel_idx] = 0;
    for (size_t c_idx = 0; c_idx < im_channels; c_idx++) {
      world_data_sum[pixel_idx * im_channels + c_idx] = 0;
    }
  }

  for (int im_idx = 0; im_idx < n_images; im_idx++) {
    printf("warpImageArray(): warping image %d\n", im_idx);

    std::vector<Eigen::Vector2f> world_location;
    std::vector<Eigen::Vector2f> im_location;


    images[im_idx]->loadImage();
    images[im_idx]->resize(warp_scale);

    im_width  = images[im_idx]->width();
    im_height = images[im_idx]->height();

    computeWarpImageMapping(im_width, im_height,
                            world_min_x, world_min_y,
                            world_max_x, world_max_y,
                            poses[im_idx],
                            world_location,
                            im_location);


    uchar* im_data = images[im_idx]->data();

    for (size_t loc_idx = 0; loc_idx < world_location.size(); loc_idx++) {
      int world_x = world_location[loc_idx](0) - world_min_x;
      int world_y = world_location[loc_idx](1) - world_min_y;
      int im_x = round(im_location[loc_idx](0));
      int im_y = round(im_location[loc_idx](1));

      if (im_x >= 0 && im_y >= 0 && im_x < im_width && im_y < im_height) {
        world_pixel_weight[world_y * world_width + world_x] += 1;

        for (size_t c_idx = 0; c_idx < im_channels; c_idx++) {
          world_data_sum[(world_y * world_width + world_x) * im_channels + c_idx] +=
                        (int)(im_data[(im_y * im_width + im_x) * im_channels + c_idx]);
        }
      }

    }
    images[im_idx]->release();
  }


  Image* world_image = new Image(world_width, world_height, im_channels);
  uchar* world_data = world_image->data();

  for (size_t pixel_idx = 0; pixel_idx < world_width*world_height; pixel_idx++) {
    for (size_t c_idx = 0; c_idx < im_channels; c_idx++) {
      world_data[pixel_idx * im_channels + c_idx] = 
                                    (uchar)((double)world_data_sum[pixel_idx * im_channels + c_idx] / 
                                            (double)(world_pixel_weight[pixel_idx]));
    }
  }

  Image* world_image_weight = new Image(world_width, world_height, 1);
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



Image* gaussianBlur(Image* img, float win_size, float sigma) {
  cv::Mat cv_mat = img->convertToCvMat();
  cv::Mat out_mat;
  cv::GaussianBlur(cv_mat, out_mat, cv::Size(win_size, win_size), sigma);

  size_t h = img->height();
  size_t w = img->width();
  size_t c = img->channels(); 
  Image* out = new Image(w, h, c);
  memcpy(out->data(), out_mat.data, h*w*c);
  return out;
}

void gray2jet(float v,float vmin, float vmax,
              float& r, float& g, float& b) {
  r = 1.0f;
  g = 1.0f;
  b = 1.0f;
  float dv;

  if (v < vmin)
    v = vmin;
  if (v > vmax)
    v = vmax;
  dv = vmax - vmin;

  if (v < (vmin + 0.25 * dv)) {
    r = 0;
    g = 4 * (v - vmin) / dv;
  } else if (v < (vmin + 0.5 * dv)) {
    r = 0;
    b = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
  } else if (v < (vmin + 0.75 * dv)) {
    r = 4 * (v - vmin - 0.5 * dv) / dv;
    b = 0;
  } else {
    g = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
    b = 0;
  }
}

// Image* grayImage2JetImage(Image* img, float vmin, float vmax) {
//   if (img->channels() == 3) {
//     img->bgr2gray();
//   } else if (img->channels() != 1) {
//     printf("grayImage2JetImage: invalid number of channels\n");
//     exit(-1);
//   }

//   size_t h = img->height();
//   size_t w = img->width();
//   Image* jet_img = new Image(w, h, 3);

//   float r, g, b;
//   for (size_t y = 0; y < h; y++) {
//     for (size_t x = 0; x < w; x++) {
//       gray2jet(img->getPixel(y, x), vmin, vmax, r, g, b);
//       jet_img->getPixel(y, x, 0) = (uchar)r * 255.0f;
//       jet_img->getPixel(y, x, 1) = (uchar)g * 255.0f;
//       jet_img->getPixel(y, x, 2) = (uchar)b * 255.0f;
//     }
//   }
//   return jet_img;
// }




}
}
