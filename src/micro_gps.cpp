#include "micro_gps.h"
#include "util.h"

#define IMAGE_ARRAY_SIZE m_database->getDatabaseSize()
// #define IMAGE_ARRAY_SIZE 10

MicroGPS::MicroGPS() {
  m_grid_step = 50.0f;
  m_flann_kdtree = NULL;
}

MicroGPS::~MicroGPS() {

}

void MicroGPS::setVotingCellSize(float m_grid_step_) {
  m_grid_step = m_grid_step_;
}

void MicroGPS::setNumScaleGroups(int num_scale_groups) {
  m_num_scale_groups = num_scale_groups;
}


void MicroGPS::loadDatabaseOnly (Database* database) {
  m_database = database;

  int num_images_to_process = IMAGE_ARRAY_SIZE;

  Eigen::MatrixXf image_locations(2, num_images_to_process);
  m_database_images.resize(num_images_to_process);

  int cnt = 0;
  for (int i = 0; i < num_images_to_process; i++) {
    WorkImage* work_image = new WorkImage(m_database->getDatabaseImage(i),
                                          m_database->getDatabasePrecomputedFeatures(i));
    
    Eigen::Matrix3f image_pose = m_database->getDatabasePose(i);
    // buffer image location
    image_locations.block(0, i, 2, 1) = image_pose.block(0, 2, 2, 1);
    m_database_images[i] = work_image;
  }
  m_database_images[0]->loadImage();
  m_image_width = m_database_images[0]->width();
  m_image_height = m_database_images[0]->height();
  printf("image size: %d x %d\n", m_image_width, m_image_height);
  float radius = sqrt((float)(m_image_width * m_image_width + m_image_height * m_image_height));
  m_database_images[0]->release();

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

  printf("MicroGPS::loadDatabaseOnly(): built grid %d x %d\n", m_grid_width, m_grid_height);
  printf("MicroGPS::loadDatabaseOnly(): grid range: [%f, %f, %f, %f]\n", m_grid_min_x, m_grid_min_y,
                                            m_grid_min_x + m_grid_width * m_grid_step,
                                            m_grid_min_y + m_grid_height * m_grid_step);

}

void MicroGPS::removeDuplicates () {
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
      Eigen::Matrix3f im_pose_i = m_database->getDatabasePose(m_feature_image_idx[i]);
      Eigen::Matrix3f im_pose_j = m_database->getDatabasePose(m_feature_image_idx[j]);
      if ((m_feature_poses[j].col(2) - m_feature_poses[i].col(2)).norm() < 8.0f && 
          (m_features.row(i) - m_features.row(j)).norm() < 0.6f) { // threshold is hard coded for siftgpu
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
  std::vector<bool> to_discard(cnt, false);

  for (int i = 0; i < selected_idx.size(); i++) {
    features_shrinked.row(i) = m_features.row(selected_idx[i]); 
    feature_poses_shrinked[i] = m_feature_poses[selected_idx[i]];
    feature_image_idx_shrinked[i] = m_feature_image_idx[selected_idx[i]];
    feature_scales_shrinked[i] = m_feature_scales[selected_idx[i]];
  }


  m_feature_poses = feature_poses_shrinked;
  m_feature_image_idx = feature_image_idx_shrinked;
  m_feature_scales = feature_scales_shrinked;
  m_features = features_shrinked;

}

void MicroGPS::computePCABasis() {
  Eigen::MatrixXf mean_deducted = m_features.rowwise() - m_features.colwise().mean();

  Eigen::JacobiSVD<Eigen::MatrixXf> svd(mean_deducted, Eigen::ComputeThinV);

  m_PCA_basis = svd.matrixV();

  printf("MicroGPS::PCAreduction(): PCA basis size: %ld x %ld\n", m_PCA_basis.rows(), m_PCA_basis.cols());
}

void MicroGPS::PCAreduction(int num_dimensions_to_keep) {
  printf("MicroGPS::PCAreduction(): PCA basis size: %ld x %ld\n", m_PCA_basis.rows(), m_PCA_basis.cols());

  Eigen::MatrixXf PCA_basis_k_cols = m_PCA_basis.leftCols(num_dimensions_to_keep);

  m_features_PCAed = m_features * PCA_basis_k_cols;
  printf("MicroGPS::PCAreduction(): m_features_PCAed size: %ld x %ld\n", m_features_PCAed.rows(), m_features_PCAed.cols());
}



void MicroGPS::preprocessDatabaseImages (int num_samples_per_image, float image_scale_for_sift) {

  int max_num_features = num_samples_per_image * m_database->getDatabaseSize();
  m_features = Eigen::MatrixXf(max_num_features, 128);
  m_feature_poses.resize(max_num_features);
  m_feature_image_idx.resize(max_num_features);
  m_feature_scales.resize(max_num_features);
  m_feature_local_location.resize(max_num_features, 4);
  
  int cnt = 0;
  for (int i = 0; i < m_database_images.size(); i++) {
    WorkImage* work_image = m_database_images[i];
    work_image->loadImage();
    if (!work_image->loadPrecomputedFeature(image_scale_for_sift)) { // prefer using precomputed features
      work_image->extractSIFT(image_scale_for_sift);
    }
    Eigen::Matrix3f image_pose = m_database->getDatabasePose(i);

    std::vector<int> sel;
    randomSample(work_image->getSIFTSize(), num_samples_per_image, sel);

    for (int j = 0; j < sel.size(); j++) {
      SIFTFeature* f = work_image->getSIFTFeature(sel[j]);
      m_features.row(cnt) = f->descriptor;
      f->global_pose = image_pose * f->local_pose;

      m_feature_poses[cnt] = f->global_pose;
      m_feature_image_idx[cnt] = i;
      m_feature_scales[cnt] = f->scale;
      m_feature_local_location(cnt, 0) = f->x;
      m_feature_local_location(cnt, 1) = f->y;
      m_feature_local_location(cnt, 2) = f->scale;
      m_feature_local_location(cnt, 3) = f->angle;
      cnt++;
    }

    work_image->release();
  }

  m_feature_poses.resize(cnt);
  m_feature_image_idx.resize(cnt);
  m_feature_scales.resize(cnt);
  m_features.conservativeResize(cnt, 128);
  m_feature_local_location.conservativeResize(cnt, 4);

  removeDuplicates();

  printf("MicroGPS::preprocessDatabase(): removed %d duplicated features\n", cnt - m_features.rows());

  printf("MicroGPS::preprocessDatabase(): m_features size: %ld x %ld\n", m_features.rows(), m_features.cols());
}


void MicroGPS::buildSearchIndex() {
  bool index_built = m_flann_kdtree != NULL;

  if (index_built) {
    delete[] m_features_PCAed_flann.ptr();
  }
  // Build flann index
  m_features_PCAed_flann = flann::Matrix<float>(new float[m_features_PCAed.rows() * m_features_PCAed.cols()],
                                                m_features_PCAed.rows(),
                                                m_features_PCAed.cols());

  // copy data
  for (int i = 0; i < m_features_PCAed.rows(); i++) {
    for (int j = 0; j < m_features_PCAed.cols(); j++) {
      m_features_PCAed_flann[i][j] = m_features_PCAed(i, j);
      // m_features_PCAed_flann[i][j] = m_features(i, j);
    }
  }
  // build kd tree
  if (index_built) {
    delete m_flann_kdtree;
    m_flann_kdtree = NULL;
  }

  // printf("building CUDA search index\n");
  // flann::KDTreeCuda3dIndexParams params;
  flann::KDTreeIndexParams params;

  m_flann_kdtree = new flann::Index<L2<float> >(m_features_PCAed_flann, params);
  m_flann_kdtree->buildIndex();
}

void MicroGPS::searchNearestNeighbors(WorkImage* work_image, std::vector<int>& nn_index) {
  int num_test_features = work_image->getSIFTSize();
  int num_dimensions_to_keep = m_PCA_basis.cols();

  flann::Matrix<float> flann_query(new float[num_test_features * num_dimensions_to_keep],
                                            num_test_features, num_dimensions_to_keep);
  flann::Matrix<int> flann_index(new int[num_test_features], num_test_features, 1);
  flann::Matrix<float> flann_dist(new float[num_test_features], num_test_features, 1);

  for (int i = 0; i < num_test_features; i++) {
    SIFTFeature* f = work_image->getSIFTFeature(i);
    for (int j = 0; j < num_dimensions_to_keep; j++) {
      flann_query[i][j] = f->descriptor_PCAed[j];
    }
  }
  
  m_flann_kdtree->knnSearch(flann_query, flann_index, flann_dist, 1, SearchParams(64));

  nn_index.resize(num_test_features);

  for (int i = 0; i < num_test_features; i++) {
    nn_index[i] = flann_index[i][0];
  }

  delete[] flann_query.ptr();
  delete[] flann_index.ptr();
  delete[] flann_dist.ptr();
}


void MicroGPS::buildSearchIndexMultiScales() {
  int m_num_search_index_ranges = m_num_scale_groups;
  
  // check if built
  bool index_built = m_flann_kdtree_multi_scales.size() > 0;
  
  if (index_built) {
    printf("Index built, deleting built index...\n");
    for (int i = 0; i < m_flann_kdtree_multi_scales.size(); i++) {
      delete[] m_features_PCAed_flann_multi_scales[i].ptr();
      delete m_flann_kdtree_multi_scales[i]; 
    }
    m_features_PCAed_flann_multi_scales.clear();
    m_flann_kdtree_multi_scales.clear();
    m_bounds_multi_scales.clear();
    m_global_index_multi_scales.clear();
  }
  
  printf("Start to build new multi-scale search index\n");

  // sort and figure out ranges
  std::vector<float> feature_scales_sorted = m_feature_scales;
  printf("%ld\n", m_feature_scales.size());
  std::sort(feature_scales_sorted.begin(), feature_scales_sorted.end()); // ascending
  int bin_size = feature_scales_sorted.size() / m_num_search_index_ranges;

  m_bounds_multi_scales.resize(m_num_search_index_ranges+1);
  m_bounds_multi_scales[0] = -1.0f; // min
  m_bounds_multi_scales[m_num_search_index_ranges] = feature_scales_sorted.back(); // max
  for (int i = 1; i < m_num_search_index_ranges; i++) {
    m_bounds_multi_scales[i] = feature_scales_sorted[bin_size * i - 1];
  }

  for (int i = 0; i < m_num_search_index_ranges+1; i++) {
    printf("%f ", m_bounds_multi_scales[i]);
  }
  printf("\n");

  std::vector<int> bin_count(m_num_search_index_ranges, 0);
  std::vector<int> bin_assignment(m_feature_scales.size(), -1);

  for (size_t i = 0; i < m_feature_scales.size(); i++) {
    float scale = m_feature_scales[i];
    for (int b = 0; b < m_num_search_index_ranges; b++) {
      if (scale > m_bounds_multi_scales[b] && scale <= m_bounds_multi_scales[b+1]) {
        bin_count[b]++;
        bin_assignment[i] = b;
        break;
      }

    }
  }


  printf("bin count\n");
  for (int i = 0; i < m_num_search_index_ranges; i++) {
    printf("%d ", bin_count[i]);
  }
  printf("\n");

  
  // allocate flann memory and copy data
  m_features_PCAed_flann_multi_scales.resize(m_num_search_index_ranges);
  for (int i = 0; i < m_num_search_index_ranges; i++) {
    if (bin_count[i] == 0) {
      continue;
    }
    m_features_PCAed_flann_multi_scales[i] = flann::Matrix<float>(new float[bin_count[i] * m_features_PCAed.cols()],   
                                                                  bin_count[i],
                                                                  m_features_PCAed.cols());
  }

  std::vector<int> bin_counter(m_num_search_index_ranges, 0);
  m_global_index_multi_scales.resize(m_num_search_index_ranges);
  for (int i = 0; i < m_num_search_index_ranges; i++) {
    m_global_index_multi_scales[i].resize(bin_count[i]);
  }
  for (size_t i = 0; i < m_features_PCAed.rows(); i++) {
    int bin_index = bin_assignment[i];
    for (size_t j = 0; j < m_features_PCAed.cols(); j++) {
      m_features_PCAed_flann_multi_scales[bin_index][bin_counter[bin_index]][j] = m_features_PCAed(i, j);
    }
    m_global_index_multi_scales[bin_index][bin_counter[bin_index]] = i;
    bin_counter[bin_index]++;
  }

  // build kd-trees
  m_flann_kdtree_multi_scales.resize(m_num_search_index_ranges);

  for (int i = 0; i < m_num_search_index_ranges; i++) {
    if (bin_count[i] == 0) {
      m_flann_kdtree_multi_scales[i] = NULL;
      continue;
    }
    m_flann_kdtree_multi_scales[i] = new flann::Index<L2<float> >(m_features_PCAed_flann_multi_scales[i], flann::KDTreeIndexParams());
    m_flann_kdtree_multi_scales[i]->buildIndex();
  }


}


void MicroGPS::searchNearestNeighborsMultiScales(WorkImage* work_image, std::vector<int>& nn_index, int best_k) {
  int num_test_features = work_image->getSIFTSize();
  int num_dimensions_to_keep = m_PCA_basis.cols();
  int num_scales = m_features_PCAed_flann_multi_scales.size();

  std::vector<flann::Matrix<float> > flann_query_multi_scales(num_scales);
  std::vector<flann::Matrix<int> > flann_index_multi_scales(num_scales);
  std::vector<flann::Matrix<float> > flann_dist_multi_scales(num_scales);

  // bin count
  std::vector<int> bin_count(num_scales, 0);
  std::vector<int> bin_assignment(num_test_features, -1);
  for (int i = 0; i < num_test_features; i++) {
    float scale = work_image->getSIFTFeature(i)->scale;
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
    SIFTFeature* f = work_image->getSIFTFeature(i);
    int bin_index = bin_assignment[i];
    if (bin_index >= 0 ) {
      for (size_t j = 0; j < num_dimensions_to_keep; j++) {
        flann_query_multi_scales[bin_index][bin_counter[bin_index]][j] = f->descriptor_PCAed[j];
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


  if (best_k > num_test_features) {
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
  for (int k = best_k; k < num_test_features; k++) {
    nn_index[sort_idx[k]] = -1;
  }
}


void MicroGPS::savePCABasis(const char* path) {
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

void MicroGPS::loadPCABasis(const char* path) {
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


void MicroGPS::saveFeatures(const char* path) {
  FILE* fp = fopen(path, "w");

  // write size
  size_t size[2];
  // size[0] = m_features_PCAed.rows();
  // size[1] = m_features_PCAed.cols();

  // // write data
  // fwrite(size, sizeof(size_t), 2, fp);
  // fwrite(m_features_PCAed.data(), sizeof(float), m_features_PCAed.cols() * m_features_PCAed.rows(), fp);

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
  fwrite(m_feature_scales.data(), sizeof(float), m_feature_image_idx.size(), fp);

  // write corresponding image index
  fwrite(m_feature_image_idx.data(), sizeof(int), m_feature_image_idx.size(), fp);

  fwrite(m_feature_local_location.data(), sizeof(float), m_feature_local_location.cols() * m_feature_local_location.rows(), fp);

  fclose(fp);
}

void MicroGPS::loadFeatures(const char* path) {
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
  m_feature_image_idx.clear();
  m_feature_image_idx.resize(size[0]);
  fread(m_feature_image_idx.data(), sizeof(int), m_feature_image_idx.size(), fp);

  fclose(fp);

  printf("MicroGPS::loadFeatures(): loaded features size: %ld x %ld\n", size[0], size[1]);

}


bool MicroGPS::locate(WorkImage* work_image, WorkImage*& alignment_image,
                      MicroGPSOptions& options,
                      MicroGPSResult& result,
                      MicroGPSTiming& timing, 
                      MicroGPSDebug& debug) {

  clock_t begin_global, end_global;
  clock_t begin_local, end_local;

  begin_global = clock();

  // assume work_image is loaded
  begin_local = clock();
  if (!work_image->loadPrecomputedFeature(options.image_scale_for_sift)) { // prefer using precomputed features
    work_image->extractSIFT(options.image_scale_for_sift);
  }
  end_local = clock();
  timing.sift_extraction = double(end_local - begin_local) / CLOCKS_PER_SEC;
  printf("sift extraction costs %f ms\n", double(end_local - begin_local) / CLOCKS_PER_SEC);


  begin_local = clock();
  work_image->PCADimReduction(m_PCA_basis);
  end_local = clock();
  // timing.pca_reduction = double(end_local - begin_local) / CLOCKS_PER_SEC);

  int num_test_features = work_image->getSIFTSize();
  int num_dimensions_to_keep = m_PCA_basis.cols();

  printf("MicroGPS::locate(): start KNN searching\n");
  begin_local = clock();


  std::vector<int> nn_index;
  // searchNearestNeighbors(work_image, nn_index);
  printf("searching for %d best nn\n", options.best_knn);
  searchNearestNeighborsMultiScales(work_image, nn_index, options.best_knn);


  end_local = clock();
  timing.knn_search = double(end_local - begin_local) / CLOCKS_PER_SEC;
  printf("knn costs %f ms\n", double(end_local - begin_local) / CLOCKS_PER_SEC);

  // compute image pose for each match
  // T_WItest = T_WIdata * T_IdataFdata * T_FdataFtest * T_FtestItest
  // T_FdataFtest = I

  begin_local = clock();
  std::vector<Eigen::Matrix3f> pose_candidates(num_test_features);
  int num_valid_NN = 0;
  for (int i = 0; i < num_test_features; i++) {
    SIFTFeature* f = work_image->getSIFTFeature(i);
    // printf("nn = %d\n", flann_index[i][0]);
    if (nn_index[i] < 0) {
      continue;
    }
    pose_candidates[i] = m_feature_poses[nn_index[i]] * f->local_pose.inverse();
    num_valid_NN++;
    // std::cout << pose_candidates[i] << std::endl << std::endl;
    // printf("nn = %d\n", nn_index[i]);
  }
  end_local = clock();
  timing.candidate_image_pose = double(end_local - begin_local) / CLOCKS_PER_SEC;
  printf("MicroGPS::locate(): computed candidate image poses\n");
  printf("compute candidate poses costs %f ms\n", double(end_local - begin_local) / CLOCKS_PER_SEC);

  // vote for the image origin
  begin_local = clock();
  m_voting_grid = std::vector<int>(m_voting_grid.size(), 0); // reset
  printf("m_grid_width = %d, m_grid_height = %d\n", m_grid_width, m_grid_height);
  for (int i = 0; i < num_test_features; i++) {
    if (nn_index[i] < 0) {
      continue;
    } 
    int cell_x = floor((pose_candidates[i](0, 2) - m_grid_min_x) / m_grid_step);
    int cell_y = floor((pose_candidates[i](1, 2) - m_grid_min_y) / m_grid_step);

    // printf("cell_x = %d, cell_y = %d\n", cell_x, cell_y);
    m_voting_grid[cell_y * m_grid_width + cell_x] += 1;
  }

  printf("MicroGPS::locate(): finished voting\n");

  //select the peak, figure out inliers
  int peak_cnt = 0;
  int peak_x, peak_y;
  for (int y = 0; y < m_grid_height; y++) {
    for (int x = 0; x < m_grid_width; x++) {
      if (m_voting_grid[y * m_grid_width + x] > peak_cnt) {
        peak_cnt = m_voting_grid[y * m_grid_width + x];
        peak_x = x;
        peak_y = y;
      }
    }
  }
  printf("MicroGPS::locate(): finished selecting the peak: (%d, %d)\n", peak_x, peak_y);

  printf("MicroGPS::locate(): top 10 cells\n");
  std::sort(m_voting_grid.begin(), m_voting_grid.end(), std::greater<int>());
  for (int i = 0; i < 10; i++) {
    printf("%d\n", m_voting_grid[i]);
  }
  result.top_cells = std::vector<int>(m_voting_grid.begin(), m_voting_grid.begin()+10);

  std::vector<int> peak_idx;
  for (int i = 0; i < num_test_features; i++) {
    // printf("(%f, %f)\n", floor((pose_candidates[i](0, 2) - m_grid_min_x) / m_grid_step),
    //                     floor((pose_candidates[i](1, 2) - m_grid_min_y) / m_grid_step));
    if (nn_index[i] < 0) {
      continue;
    }
    if (floor((pose_candidates[i](0, 2) - m_grid_min_x) / m_grid_step) == peak_x &&
        floor((pose_candidates[i](1, 2) - m_grid_min_y) / m_grid_step) == peak_y) {
      peak_idx.push_back(i);
    }
  }
  printf("MicroGPS::locate(): peak size = %ld\n", peak_idx.size());
  end_local = clock();
  timing.voting = double(end_local - begin_local) / CLOCKS_PER_SEC;
  printf("voting costs %f ms\n", double(end_local - begin_local) / CLOCKS_PER_SEC);

  if (peak_idx.size() < 2) {
    printf("MicroGPS::locate(): return false because peak_idx.size() < 2\n");
    return false; // pose cannot be determined
  }

  //re-estimate pose using inliers
  begin_local = clock();
  Eigen::MatrixXf points1(peak_idx.size(), 2);
  Eigen::MatrixXf points2(peak_idx.size(), 2);
  // std::vector<int> nearest_database_images_idx(peak_idx.size()); // for verification purpose

  for (int i = 0; i < peak_idx.size(); i++) {
    SIFTFeature* f_test = work_image->getSIFTFeature(peak_idx[i]);
    points1(i, 0) = f_test->x;
    points1(i, 1) = f_test->y;

    int f_database_idx = nn_index[peak_idx[i]];
    points2(i, 0) = m_feature_poses[f_database_idx](0, 2);
    points2(i, 1) = m_feature_poses[f_database_idx](1, 2);
    // nearest_database_images_idx[i] = m_feature_image_idx[f_database_idx];

    printf("MicroGPS::locate(): peak feature is from image %d\n", m_feature_image_idx[f_database_idx]);
  }


  Eigen::MatrixXf pose_estimated;
  std::vector<int> ransac_inliers;
  estimateRigidTransformation(points1, points2,
                              pose_estimated, ransac_inliers,
                              1000, 5.0f);

  printf("MicroGPS::locate(): RANSAC inliers %ld\n", ransac_inliers.size());

  if (ransac_inliers.size() < 2) {
    printf("MicroGPS::locate(): RANSAC not enough inliers: %ld\n", ransac_inliers.size());
    printf("MicroGPS::locate(): return false because ransac_inliers.size() < 2\n");
    return false;
  }


  // work_image->release();
  end_local = clock();
  timing.ransac = double(end_local - begin_local) / CLOCKS_PER_SEC;
  printf("ransac costs %f ms\n", double(end_local - begin_local) / CLOCKS_PER_SEC);

  end_global = clock();
  timing.total = double(end_global - begin_global) / CLOCKS_PER_SEC;
  printf("total costs %f ms\n", double(end_global - begin_global) / CLOCKS_PER_SEC);


  result.final_estimated_pose = pose_estimated;

  printf("MicroGPS::locate(): gathering debug info\n");
  // gather debug data
  if (options.debug_mode) {
    debug.grid_step = m_grid_step;
    debug.peak_topleft_x = peak_x * m_grid_step + m_grid_min_x;
    debug.peak_topleft_y = peak_y * m_grid_step + m_grid_min_y;

    debug.knn_matched_feature_poses.resize(num_valid_NN);
    debug.candidate_image_poses.resize(num_valid_NN);
    debug.test_feature_poses.resize(num_valid_NN);

    int debug_cnt = 0;
    for (int i = 0; i < num_test_features; i++) {
      SIFTFeature* f = work_image->getSIFTFeature(i);
      if (nn_index[i] > 0) {
        debug.candidate_image_poses[debug_cnt] = m_feature_poses[nn_index[i]] * f->local_pose.inverse();
        debug.knn_matched_feature_poses[debug_cnt] = m_feature_poses[nn_index[i]];
        debug.test_feature_poses[debug_cnt] = f->local_pose;
        debug_cnt++;
      }
      // printf("i = %d / %d\n", i, num_test_features);
    }
  }

  printf("MicroGPS::locate(): start verification\n");
  // --------------- VERIFICATION ----------------------
  if (!options.do_alignment && !options.do_siftmatch) {
    // TODO: compute confidence and return
    alignment_image = NULL;
    return true;
  }
  
  double rect_verts_x[4];
  double rect_verts_y[4];
  double* m_feature_poses_x = new double[m_feature_image_idx.size()];
  double* m_feature_poses_y = new double[m_feature_image_idx.size()];
  bool* points_in_on = new bool[m_feature_image_idx.size()];
  bool* points_in = new bool[m_feature_image_idx.size()];
  bool* points_on = new bool[m_feature_image_idx.size()];

  float lx = 0;
  float ly = 0;
  float ux = work_image->width()-1;
  float uy = work_image->height()-1;
  Eigen::MatrixXf corners(2, 4);
  corners(0, 0) = lx;
  corners(1, 0) = ly;
  corners(0, 1) = lx;
  corners(1, 1) = uy;
  corners(0, 2) = ux;
  corners(1, 2) = uy;
  corners(0, 3) = ux;
  corners(1, 3) = ly;
  corners = pose_estimated.block(0, 0, 2, 2) * corners;
  corners.row(0) = corners.row(0).array() + pose_estimated(0, 2);
  corners.row(1) = corners.row(1).array() + pose_estimated(1, 2);
  
  for (int vert_idx = 0; vert_idx < 4; vert_idx++) {
    rect_verts_x[vert_idx] = corners.row(0)(vert_idx);
    rect_verts_y[vert_idx] = corners.row(1)(vert_idx);
  }
  
  for (int f_idx = 0; f_idx < m_feature_poses.size(); f_idx++) {
    m_feature_poses_x[f_idx] = m_feature_poses[f_idx](0, 2);
    m_feature_poses_y[f_idx] = m_feature_poses[f_idx](1, 2);
  }
  
  inpolygon(m_feature_image_idx.size(), m_feature_poses_x, m_feature_poses_y,  
            rect_verts_x,  rect_verts_y,  points_in_on,  points_in, points_on);
  
  
  std::vector<int> database_image_relevance(m_feature_image_idx.size(), 0); // for verification purpose
  for (int f_idx = 0; f_idx < m_feature_poses.size(); f_idx++) {
    if (points_in_on[f_idx]) {
      database_image_relevance[m_feature_image_idx[f_idx]]++;
      // printf("feature from image %d falls in the rectangle\n", m_feature_image_idx[f_idx]);
    }
  }
  delete[] m_feature_poses_x;
  delete[] m_feature_poses_y;
  delete[] points_in_on;
  delete[] points_in;
  delete[] points_on; 

  // get the most relevant database image index
  int closest_database_image_idx = -1;
  int max_relevance = 0;
  for (int im_idx = 0; im_idx < database_image_relevance.size(); im_idx++) {
    if (database_image_relevance[im_idx] > max_relevance) {
      max_relevance = database_image_relevance[im_idx];
      closest_database_image_idx = im_idx;
    }
  }
  printf("max_relevance = %d\n", max_relevance);
  debug.closest_database_image_idx = closest_database_image_idx;

  //SIFT-RANSAC verification
  if (options.do_siftmatch) {
    m_database_images[closest_database_image_idx]->loadImage();
    // work_image->loadImage();

    std::vector<int> matched_idx1;
    std::vector<int> matched_idx2;
    work_image->siftMatch(m_database_images[closest_database_image_idx], matched_idx1, matched_idx2);

    if (matched_idx1.size() < 5) {
      printf("MicroGPS::locate(): return false because verification matched_idx1.size() < 5\n");
      m_database_images[closest_database_image_idx]->release();
      return false;
    }

    Eigen::MatrixXf pose_verified;
    std::vector<int> inliers_verified;
    // work_image->siftMatchEstimatePose(m_database_images[closest_database_image_idx], matched_idx1, matched_idx2,
    //                                               pose_verified, inliers_verified);
    m_database_images[closest_database_image_idx]->siftMatchEstimatePose(work_image, matched_idx2, matched_idx1,
                                                                        pose_verified, inliers_verified);

    if (inliers_verified.size() < 5) {
      printf("MicroGPS::locate(): return false because verification inliers_verified.size() < 5\n");
      m_database_images[closest_database_image_idx]->release();
      return false;
    }
    m_database_images[closest_database_image_idx]->release();
    // printf("num test features = %ld\n", debug.test_feature_poses.size()); 
    
    result.siftmatch_estimated_pose = m_database->getDatabasePose(closest_database_image_idx) * pose_verified;
    debug.test_image_path = std::string(work_image->getImagePath());
    debug.closest_database_image_path = std::string(m_database_images[closest_database_image_idx]->getImagePath());
  
    result.x_error = pose_estimated(0, 2) - result.siftmatch_estimated_pose(0, 2);
    result.y_error = pose_estimated(1, 2) - result.siftmatch_estimated_pose(1, 2);
    float angle_estimated = atan2(pose_estimated(0, 1), pose_estimated(0, 0)) / M_PI * 180.0f;
    float angle_verified = atan2(result.siftmatch_estimated_pose(0, 1), result.siftmatch_estimated_pose(0, 0)) / M_PI * 180.0f;
    result.angle_error = angle_estimated - angle_verified;
  }

  if (options.do_alignment) {
    //stitch images to verify
    std::vector<WorkImage*> image_array;
    image_array.push_back(m_database_images[closest_database_image_idx]);
    image_array.push_back(work_image);
    std::vector<Eigen::Matrix3f> pose_array;
    pose_array.push_back(m_database->getDatabasePose(closest_database_image_idx));
    pose_array.push_back(pose_estimated);
    // pose_array.push_back(result.siftmatch_estimated_pose);
    alignment_image = warpImageArray(image_array, pose_array, 0.25);
    // alignment_image->write("align.png");
  }

  return true;
}