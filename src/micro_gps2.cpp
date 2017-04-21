#include "micro_gps2.h"
#include "util2.h"

namespace MicroGPS {

Localization::Localization() {
  m_grid_step = 50.0f;
}

Localization::~Localization() {

}

void Localization::setVotingCellSize(const float cell_size) {
  m_grid_step = cell_size;
}

void Localization::setNumScaleGroups(const int num_scale_groups) {
  m_num_scale_groups = num_scale_groups;
}

void Localization::loadImageDataset(MicroGPS::ImageDataset* image_dataset) {
  m_image_dataset = image_dataset;

  int num_images_to_process = m_image_dataset->getDatabaseSize();

  Eigen::MatrixXf image_locations(2, num_images_to_process);
  m_database_images.resize(num_images_to_process);

  for (int i = 0; i < num_images_to_process; i++) {
    char precomputed_feat_path[256];
    char precomputed_sift_path[256];

    m_image_dataset->getDatabaseImagePrecomputedFeatures(i, precomputed_feat_path);
    m_image_dataset->getDatabaseImagePrecomputedFeatures(i, precomputed_sift_path, (const char*)("sift"));
    
    // printf("image_path = %s\n", m_image_dataset->getDatabaseImagePath(i));
    MicroGPS::Image* work_image = new MicroGPS::Image(m_image_dataset->getDatabaseImagePath(i),
                                                      precomputed_feat_path,
                                                      precomputed_sift_path);

    Eigen::Matrix3f image_pose = m_image_dataset->getDatabaseImagePose(i);
   
    // buffer image location
    image_locations.block(0, i, 2, 1) = image_pose.block(0, 2, 2, 1);
    m_database_images[i] = work_image;
  }

  // get database image size  
  m_database_images[0]->loadImage();
  m_image_width = m_database_images[0]->width();
  m_image_height = m_database_images[0]->height();
  printf("Database image size: %d x %d\n", m_image_width, m_image_height);
  float radius = sqrt((float)(m_image_width * m_image_width + m_image_height * m_image_height));
  m_database_images[0]->release();

  // compute boundaries for the voting map
  Eigen::MatrixXf upperbound = image_locations.rowwise().maxCoeff();
  Eigen::MatrixXf lowerbound = image_locations.rowwise().minCoeff();
  upperbound(0) += (2 * radius + m_grid_step / 2.0f);
  upperbound(1) += (2 * radius + m_grid_step / 2.0f);
  lowerbound(0) -= (2 * radius + m_grid_step / 2.0f);
  lowerbound(1) -= (2 * radius + m_grid_step / 2.0f);

  // precompute grid
  m_grid_width = ceil((upperbound(0) - lowerbound(0)) / m_grid_step);
  m_grid_height = ceil((upperbound(1) - lowerbound(1)) / m_grid_step);
  m_grid_min_x = lowerbound(0);
  m_grid_min_y = lowerbound(1);
  m_voting_grid.resize(m_grid_width * m_grid_height, 0);

  printf("loadImageDataset(): built grid %d x %d\n", m_grid_width, m_grid_height);
  printf("loadImageDataset(): grid range: [%f, %f, %f, %f]\n", 
                                            m_grid_min_x, m_grid_min_y,
                                            m_grid_min_x + m_grid_width * m_grid_step,
                                            m_grid_min_y + m_grid_height * m_grid_step);

}



void Localization::computePCABasis() {
  Eigen::MatrixXf mean_deducted = m_features.rowwise() - m_features.colwise().mean();

  Eigen::JacobiSVD<Eigen::MatrixXf> svd(mean_deducted, Eigen::ComputeThinV);

  m_PCA_basis = svd.matrixV();

  printf("computePCABasis(): PCA basis size: %ld x %ld\n", m_PCA_basis.rows(), m_PCA_basis.cols());
}


void Localization::dimensionReductionPCA(const int num_dimensions_to_keep) {
  Eigen::MatrixXf PCA_basis_k_cols = m_PCA_basis.leftCols(num_dimensions_to_keep);

  m_features_short = m_features * PCA_basis_k_cols;
  printf("dimensionReductionPCA(): m_features_short size: %ld x %ld\n", m_features_short.rows(), m_features_short.cols());
}


void Localization::preprocessDatabaseImages(const int num_samples_per_image, 
                                            const float image_scale_for_sift) 
{
  int max_num_features = num_samples_per_image * m_image_dataset->getDatabaseSize();
  m_features = Eigen::MatrixXf(max_num_features, 128);
  m_feature_poses.resize(max_num_features);
  m_feature_image_indices.resize(max_num_features);
  m_feature_scales.resize(max_num_features);
  m_feature_local_locations.resize(max_num_features, 4);
  
  int cnt = 0;
  for (int i = 0; i < m_database_images.size(); i++) {
    MicroGPS::Image* work_image = m_database_images[i];
    work_image->loadImage();
    if (!work_image->loadPrecomputedFeatures(false)) { // prefer using precomputed features
      work_image->extractSIFT(image_scale_for_sift);
    }
    Eigen::Matrix3f image_pose = m_image_dataset->getDatabaseImagePose(i);

    // random sample sift features
    std::vector<int> sel;
    util::randomSample(work_image->getNumLocalFeatures(), num_samples_per_image, sel);

    for (int j = 0; j < sel.size(); j++) {
      LocalFeature* f = work_image->getLocalFeature(sel[j]);
      m_features.row(cnt) = f->descriptor;
      f->global_pose = image_pose * f->local_pose;

      m_feature_poses[cnt] = f->global_pose;
      m_feature_image_indices[cnt] = i;
      m_feature_scales[cnt] = f->scale;
      m_feature_local_locations(cnt, 0) = f->x;
      m_feature_local_locations(cnt, 1) = f->y;
      m_feature_local_locations(cnt, 2) = f->scale;
      m_feature_local_locations(cnt, 3) = f->angle;
      cnt++;
    }

    work_image->release();
  }

  m_feature_poses.resize(cnt);
  m_feature_image_indices.resize(cnt);
  m_feature_scales.resize(cnt);
  m_features.conservativeResize(cnt, m_features.cols());
  m_feature_local_locations.conservativeResize(cnt, 4);

  removeDuplicatedFeatures();

  printf("preprocessDatabaseImages(): removed %ld duplicated features\n", cnt - m_features.rows());

  printf("preprocessDatabaseImages(): m_features size: %ld x %ld\n", m_features.rows(), m_features.cols());

}

void Localization::removeDuplicatedFeatures() {
  int num_features = m_features.rows(); // current number of features
  std::vector<bool> id_assigned(num_features, 0);
  std::vector<int> selected_idx(num_features, -1);

  int cnt = 0;
  for (int i = 0; i < num_features; i++) {
    if (id_assigned[i]) {
      continue;
    }
    float min_dist2center = 99999.9f;
    int min_dist2center_idx = -1;
    for (int j = 0; j < num_features; j++) {
      Eigen::Matrix3f im_pose_i = m_image_dataset->getDatabaseImagePose(m_feature_image_indices[i]);
      Eigen::Matrix3f im_pose_j = m_image_dataset->getDatabaseImagePose(m_feature_image_indices[j]);
      // TODO: threshold is hard coded for siftgpu
      if ((m_feature_poses[j].col(2) - m_feature_poses[i].col(2)).norm() < 8.0f && 
          (m_features.row(i) - m_features.row(j)).norm() < 0.6f) {
        Eigen::Vector3f center_loc;
        center_loc(0) = (float)m_image_width / 2.0f;
        center_loc(1) = (float)m_image_height / 2.0f;
        center_loc(2) = 1.0f;
        Eigen::Vector3f center_j = im_pose_j * center_loc;
        // float dist_i = (center_i - m_feature_poses[i].col(2)).norm();
        float dist2center = (center_j - m_feature_poses[j].col(2)).norm();
        if (dist2center < min_dist2center) {
          // printf("i = %d, j = %d, dist2center = %f\n", i, j, dist2center);
          min_dist2center = dist2center;
          min_dist2center_idx = j;
        }
        id_assigned[j] = true;
      }
    }
    // printf("%d\n", min_dist2center_idx);
    selected_idx[cnt] = min_dist2center_idx;
    cnt++;
  }
  // printf("num_features = %d\n", num_features);

  selected_idx.resize(cnt);

  Eigen::MatrixXf features_shrinked(cnt, 128);
  std::vector<Eigen::Matrix3f> feature_poses_shrinked(cnt);
  std::vector<int> feature_image_idx_shrinked(cnt);
  std::vector<float> feature_scales_shrinked(cnt);

  for (int i = 0; i < selected_idx.size(); i++) {
    features_shrinked.row(i) = m_features.row(selected_idx[i]); 
    feature_poses_shrinked[i] = m_feature_poses[selected_idx[i]];
    feature_image_idx_shrinked[i] = m_feature_image_indices[selected_idx[i]];
    feature_scales_shrinked[i] = m_feature_scales[selected_idx[i]];
  }


  m_feature_poses = feature_poses_shrinked;
  m_feature_image_indices = feature_image_idx_shrinked;
  m_feature_scales = feature_scales_shrinked;
  m_features = features_shrinked;

}


void Localization::buildSearchIndexMultiScales() {
  // check if built
  bool index_built = m_flann_kdtree_multi_scales.size() > 0;
  
  if (index_built) {
    printf("Index built, deleting built index...\n");
    for (int i = 0; i < m_flann_kdtree_multi_scales.size(); i++) {
      delete[] m_features_short_flann_multi_scales[i].ptr();
      delete m_flann_kdtree_multi_scales[i]; 
    }
    m_features_short_flann_multi_scales.clear();
    m_flann_kdtree_multi_scales.clear();
    m_bounds_multi_scales.clear();
    m_global_index_multi_scales.clear();
  }
  
  printf("Start to build new multi-scale search index\n");

  // sort and figure out ranges
  std::vector<float> feature_scales_sorted = m_feature_scales;
  printf("%ld\n", m_feature_scales.size());
  std::sort(feature_scales_sorted.begin(), feature_scales_sorted.end()); // ascending
  int bin_size = feature_scales_sorted.size() / m_num_scale_groups;

  m_bounds_multi_scales.resize(m_num_scale_groups+1);
  m_bounds_multi_scales[0] = -1.0f; // min
  m_bounds_multi_scales[m_num_scale_groups] = feature_scales_sorted.back(); // max
  for (int i = 1; i < m_num_scale_groups; i++) {
    m_bounds_multi_scales[i] = feature_scales_sorted[bin_size * i - 1];
  }

  for (int i = 0; i < m_num_scale_groups+1; i++) {
    printf("%f ", m_bounds_multi_scales[i]);
  }
  printf("\n");

  std::vector<int> bin_count(m_num_scale_groups, 0);
  std::vector<int> bin_assignment(m_feature_scales.size(), -1);

  for (size_t i = 0; i < m_feature_scales.size(); i++) {
    float scale = m_feature_scales[i];
    for (int b = 0; b < m_num_scale_groups; b++) {
      if (scale > m_bounds_multi_scales[b] && scale <= m_bounds_multi_scales[b+1]) {
        bin_count[b]++;
        bin_assignment[i] = b;
        break;
      }

    }
  }


  printf("bin count\n");
  for (int i = 0; i < m_num_scale_groups; i++) {
    printf("%d ", bin_count[i]);
  }
  printf("\n");

  
  // allocate flann memory and copy data
  m_features_short_flann_multi_scales.resize(m_num_scale_groups);
  for (int i = 0; i < m_num_scale_groups; i++) {
    if (bin_count[i] == 0) {
      continue;
    }
    m_features_short_flann_multi_scales[i] = flann::Matrix<float>(
                                                  new float[bin_count[i] * m_features_short.cols()],   
                                                  bin_count[i],
                                                  m_features_short.cols());
  }

  std::vector<int> bin_counter(m_num_scale_groups, 0);
  m_global_index_multi_scales.resize(m_num_scale_groups);
  for (int i = 0; i < m_num_scale_groups; i++) {
    m_global_index_multi_scales[i].resize(bin_count[i]);
  }
  for (size_t i = 0; i < m_features_short.rows(); i++) {
    int bin_index = bin_assignment[i];
    for (size_t j = 0; j < m_features_short.cols(); j++) {
      m_features_short_flann_multi_scales[bin_index][bin_counter[bin_index]][j] = m_features_short(i, j);
    }
    m_global_index_multi_scales[bin_index][bin_counter[bin_index]] = i;
    bin_counter[bin_index]++;
  }

  // build kd-trees
  m_flann_kdtree_multi_scales.resize(m_num_scale_groups);

  for (int i = 0; i < m_num_scale_groups; i++) {
    if (bin_count[i] == 0) {
      m_flann_kdtree_multi_scales[i] = NULL;
      continue;
    }
    m_flann_kdtree_multi_scales[i] = new flann::Index<L2<float> >(m_features_short_flann_multi_scales[i], flann::KDTreeIndexParams());
    m_flann_kdtree_multi_scales[i]->buildIndex();
  }


}

void Localization::searchNearestNeighborsMultiScales(MicroGPS::Image* work_image, 
                                        std::vector<int>& nn_index, 
                                        int best_knn)
{
  int num_test_features = work_image->getNumLocalFeatures();
  int num_dimensions_to_keep = m_PCA_basis.cols();
  int num_scales = m_features_short_flann_multi_scales.size();

  std::vector<flann::Matrix<float> > flann_query_multi_scales(num_scales);
  std::vector<flann::Matrix<int> > flann_index_multi_scales(num_scales);
  std::vector<flann::Matrix<float> > flann_dist_multi_scales(num_scales);

  // bin count
  std::vector<int> bin_count(num_scales, 0);
  std::vector<int> bin_assignment(num_test_features, -1);
  for (int i = 0; i < num_test_features; i++) {
    float scale = work_image->getLocalFeature(i)->scale;
    for (int b = 0; b < num_scales; b++) {
      if (scale > m_bounds_multi_scales[b] && scale <= m_bounds_multi_scales[b+1]) {
        bin_count[b]++;
        bin_assignment[i] = b;
        break;
      }
    }
  }
  
  for (int i = 0; i < num_scales; i++) {
    printf("%d ", bin_count[i]);
  }
  printf("\n");

  // allocate memory and copy data
  for (int i = 0; i < num_scales; i++) {
    flann_query_multi_scales[i] = flann::Matrix<float>(new float[bin_count[i] * num_dimensions_to_keep],   
                                                                  bin_count[i],
                                                                  num_dimensions_to_keep);
    flann_index_multi_scales[i] = flann::Matrix<int>(new int[bin_count[i]], bin_count[i], 1);
    flann_dist_multi_scales[i] = flann::Matrix<float>(new float[bin_count[i]], bin_count[i], 1);
  }

  std::vector<int> bin_counter(num_scales, 0);
  for (size_t i = 0; i < num_test_features; i++) {
    LocalFeature* f = work_image->getLocalFeature(i);
    int bin_index = bin_assignment[i];
    if (bin_index >= 0 ) {
      for (size_t j = 0; j < num_dimensions_to_keep; j++) {
        flann_query_multi_scales[bin_index][bin_counter[bin_index]][j] = f->descriptor_compressed[j];
      }
      bin_counter[bin_index]++;
    }
  }
  printf("flann memory allocated and copied\n");


  // knn search
  // TODO: parallelize different scales
  for (size_t i = 0; i < num_scales; i++) {
    if (!m_flann_kdtree_multi_scales[i]) {
      continue;
    }
    m_flann_kdtree_multi_scales[i]->knnSearch(flann_query_multi_scales[i], 
                                              flann_index_multi_scales[i], 
                                              flann_dist_multi_scales[i], 
                                              1, flann::SearchParams(64));
  }
  printf("flann searching done\n");

  // aggregate result
  nn_index.resize(num_test_features);
  std::vector<float> nn_dist(num_test_features);
  bin_counter = std::vector<int>(num_scales, 0);
  for (size_t i = 0; i < num_test_features; i++) {
    int bin_index = bin_assignment[i];
    if (bin_index >= 0 ) {
      nn_index[i] = m_global_index_multi_scales[bin_index][flann_index_multi_scales[bin_index][bin_counter[bin_index]][0]];
      nn_dist[i] = flann_dist_multi_scales[bin_index][bin_counter[bin_index]][0];
      bin_counter[bin_index]++;
    } else {
      nn_index[i] = -1;
      nn_dist[i] = 99999.0f;
    }
  }  

  for (int i = 0; i < num_scales; i++) {
    delete[] flann_query_multi_scales[i].ptr();
    delete[] flann_index_multi_scales[i].ptr();
    delete[] flann_dist_multi_scales[i].ptr();
  }


  if (best_knn > num_test_features) {
    return;
  }

  // sort according to distance and save the index
  std::vector<int> sort_idx(nn_dist.size());
  int n = 0;
  std::generate(std::begin(sort_idx), std::end(sort_idx), [&]{ return n++;});
  std::sort(std::begin(sort_idx), 
            std::end(sort_idx),
            [&](int i1, int i2) { return nn_dist[i1] < nn_dist[i2];});
  
  // only keep the top k
  for (int k = best_knn; k < num_test_features; k++) {
    nn_index[sort_idx[k]] = -1;
  }
}

void Localization::savePCABasis(const char* path) {
  FILE* fp = fopen(path, "w");

  // write size
  size_t size[2];
  size[0] = m_PCA_basis.rows();
  size[1] = m_PCA_basis.cols();

  // write data
  fwrite(size, sizeof(size_t), 2, fp);
  fwrite(m_PCA_basis.data(), sizeof(float), m_PCA_basis.cols() * m_PCA_basis.rows(), fp);

  fclose(fp);
}

void Localization::loadPCABasis(const char* path) {
  FILE* fp = fopen(path, "r");

  // read size
  size_t size[2];
  fread(size, sizeof(size_t), 2, fp);

  // read data
  m_PCA_basis = Eigen::MatrixXf(size[0], size[1]);
  fread(m_PCA_basis.data(), sizeof(float), size[0] * size[1], fp);

  fclose(fp);

  printf("MicroGPS::loadPCABasis(): loaded PCA basis size: %ld x %ld\n", size[0], size[1]);
}

void Localization::saveFeatures(const char* path) {
  FILE* fp = fopen(path, "w");

  // write size
  size_t size[2];

  size[0] = m_features.rows();
  size[1] = m_features.cols();

  // write data
  fwrite(size, sizeof(size_t), 2, fp);
  fwrite(m_features.data(), sizeof(float), m_features.cols() * m_features.rows(), fp);

  // write feature global poses
  for (size_t i = 0; i < m_feature_poses.size(); i++) {
    fwrite(m_feature_poses[i].data(), sizeof(float), 9, fp);
  }

  // write feature scales
  fwrite(m_feature_scales.data(), sizeof(float), m_feature_image_indices.size(), fp);

  // write corresponding image index
  fwrite(m_feature_image_indices.data(), sizeof(int), m_feature_image_indices.size(), fp);

  fwrite(m_feature_local_locations.data(), sizeof(float), m_feature_local_locations.cols() * m_feature_local_locations.rows(), fp);

  fclose(fp);
}

void Localization::loadFeatures(const char* path) {
  FILE* fp = fopen(path, "r");

  // read size
  size_t size[2];
  fread(size, sizeof(size_t), 2, fp);

  // read data
  // m_features_PCAed = Eigen::MatrixXf(size[0], size[1]);
  // fread(m_features_PCAed.data(), sizeof(float), size[0] * size[1], fp);

  m_features = Eigen::MatrixXf(size[0], size[1]);
  fread(m_features.data(), sizeof(float), size[0] * size[1], fp);

  // read feature global poses
  m_feature_poses.clear();
  m_feature_poses.resize(size[0]);
  for (size_t i = 0; i < size[0]; i++) {
    fread(m_feature_poses[i].data(), sizeof(float), 9, fp);
  }

  // read feature scales
  m_feature_scales.clear();
  m_feature_scales.resize(size[0]);
  fread(m_feature_scales.data(), sizeof(float), m_feature_scales.size(), fp);
  
  // read corresponding image index
  m_feature_image_indices.clear();
  m_feature_image_indices.resize(size[0]);
  fread(m_feature_image_indices.data(), sizeof(int), m_feature_image_indices.size(), fp);

  fclose(fp);

  printf("MicroGPS::loadFeatures(): loaded features size: %ld x %ld\n", size[0], size[1]);
}


}