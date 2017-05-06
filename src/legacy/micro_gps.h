#ifndef _MICRO_GPS_H_
#define _MICRO_GPS_H_

#include <string>
#include <Eigen/Dense>

#include "dataset.h"
#include "work_image.h"
// #define FLANN_USE_CUDA
#include "flann/flann.h"

extern "C" {
  #include "inpolygon.h"
}


struct MicroGPSTiming {
  double sift_extraction;
  double knn_search;
  double candidate_image_pose;
  double voting;
  double ransac;
  double total;
  
  MicroGPSTiming() {
    reset();
  }
  
  MicroGPSTiming& operator= (MicroGPSTiming& arg) {
    sift_extraction = arg.sift_extraction;
    knn_search = arg.knn_search;
    candidate_image_pose = arg.candidate_image_pose;
    voting = arg.voting;
    ransac = arg.ransac;
    total = arg.total;
    return *this;
  }
  
  void printToFile(FILE*& fp) {
    fprintf(fp, "----- Timing info -----\n");
    fprintf(fp, "Total: %f ms\n", total * 1000.0f);
    fprintf(fp, "Feature Extraction: %f ms\n", sift_extraction * 1000.0f);
    fprintf(fp, "NN Search: %f ms\n", knn_search * 1000.0f);
    fprintf(fp, "Compute Candidate Poses: %f ms\n", candidate_image_pose * 1000.0f);
    fprintf(fp, "Voting: %f ms\n", voting * 1000.0f);
    fprintf(fp, "RANSAC: %f ms\n", ransac * 1000.0f);
  }

  void reset() {
    sift_extraction = 0.0f;
    knn_search = 0.0f;
    candidate_image_pose = 0.0f;
    voting = 0.0f;
    ransac = 0.0f;
    total = 0.0f;
  }

};


struct MicroGPSDebug {
  float grid_step;
  float peak_topleft_x;
  float peak_topleft_y;
  std::vector<Eigen::Matrix3f> knn_matched_feature_poses; // draw pose frame
  std::vector<Eigen::Matrix3f> candidate_image_poses; // draw rect
  std::vector<Eigen::Matrix3f> test_feature_poses; // draw pose frame
  std::string test_image_path;
  std::string closest_database_image_path;
  int closest_database_image_idx;
  // std::vector<int> top_cells;
  // Eigen::MatrixXf final_estimated_pose;

  MicroGPSDebug() {
    reset();
  }
  
  MicroGPSDebug& operator= (MicroGPSDebug& arg) {
    grid_step = arg.grid_step;
    peak_topleft_x = arg.peak_topleft_x;
    peak_topleft_y = arg.peak_topleft_y;
    knn_matched_feature_poses = arg.knn_matched_feature_poses;
    candidate_image_poses = arg.candidate_image_poses;
    test_feature_poses = arg.test_feature_poses;
    test_image_path = arg.test_image_path;
    closest_database_image_path = arg.closest_database_image_path;
    closest_database_image_idx = arg.closest_database_image_idx;
    return *this;
  }

  void printToFile(FILE*& fp) {
    fprintf(fp, "----- Debug info -----\n");
    fprintf(fp, "Test image path: %s\n", test_image_path.c_str());
    fprintf(fp, "Closest database image path: %s\n", closest_database_image_path.c_str());
    fprintf(fp, "Grid step: %f\n", grid_step);
    fprintf(fp, "Peak top-left x: %f\n", peak_topleft_x);
    fprintf(fp, "Peak top-left y: %f\n", peak_topleft_y);
  }

  void reset() {
    grid_step = -1.0f;
    peak_topleft_x = -9999.0f;
    peak_topleft_y = -9999.0f;
    knn_matched_feature_poses.clear();
    candidate_image_poses.clear();
    test_feature_poses.clear();
    test_image_path = "";
    closest_database_image_path = "";
    closest_database_image_idx = -1;
  }

};

struct MicroGPSResult {
  Eigen::MatrixXf final_estimated_pose;
  float confidence;  
  bool success_flag;
  std::vector<int> top_cells;
  Eigen::MatrixXf siftmatch_estimated_pose;
  float x_error; //comparing with reult from siftmatch
  float y_error; 
  float angle_error;

  MicroGPSResult() {
    reset();
  }
  
  MicroGPSResult& operator= (MicroGPSResult& arg) {
    confidence = arg.confidence;
    success_flag = arg.success_flag;
    top_cells = arg.top_cells;
    final_estimated_pose = arg.final_estimated_pose;
    siftmatch_estimated_pose = arg.siftmatch_estimated_pose;
    x_error = arg.x_error;
    y_error = arg.y_error;
    angle_error = arg.angle_error;
    return *this;
  }
  
  void printToFile(FILE*& fp) {
    fprintf(fp, "----- Result -----\n");
    fprintf(fp, "Estimated Pose: %f %f %f %f %f %f %f %f %f\n", final_estimated_pose(0, 0), final_estimated_pose(0, 1), final_estimated_pose(0, 2),
                                                                final_estimated_pose(1, 0), final_estimated_pose(1, 1), final_estimated_pose(1, 2),
                                                                final_estimated_pose(2, 0), final_estimated_pose(2, 1), final_estimated_pose(2, 2));
    fprintf(fp, "Success: %d\n", success_flag);
    fprintf(fp, "Top cells: ");
    for(int i =0; i < top_cells.size(); i++) {
      fprintf(fp, "%d ", top_cells[i]);
    }
    fprintf(fp, "\n");

    fprintf(fp, "SIFTMatch Estimated Pose: %f %f %f %f %f %f %f %f %f\n", siftmatch_estimated_pose(0, 0), siftmatch_estimated_pose(0, 1), siftmatch_estimated_pose(0, 2),
                                                                          siftmatch_estimated_pose(1, 0), siftmatch_estimated_pose(1, 1), siftmatch_estimated_pose(1, 2),
                                                                          siftmatch_estimated_pose(2, 0), siftmatch_estimated_pose(2, 1), siftmatch_estimated_pose(2, 2));

    fprintf(fp, "x_error = %f, y_error = %f, angle_error = %f\n", x_error, y_error, angle_error);
  }

  void reset() {
    final_estimated_pose = Eigen::MatrixXf::Identity(3,3);
    siftmatch_estimated_pose = Eigen::MatrixXf::Identity(3,3);
    confidence = 0.0f;
    success_flag = false;
    top_cells.resize(0);
    x_error = -9999.0f;
    y_error = -9999.0f;
    angle_error = -9999.0f;
  }
};

struct MicroGPSOptions {
  bool do_alignment;
  bool debug_mode;
  bool do_siftmatch;
  float confidence_thresh;
  float image_scale_for_sift;
  int best_knn;

  MicroGPSOptions() {
    reset();
  }

  MicroGPSOptions& operator= (MicroGPSOptions& arg) {
    do_alignment = arg.do_alignment;
    debug_mode = arg.debug_mode;
    do_siftmatch = arg.do_siftmatch;
    confidence_thresh = arg.confidence_thresh;
    image_scale_for_sift = arg.image_scale_for_sift;
    best_knn = arg.best_knn;
    return *this;
  }

  void reset() {
    do_alignment = true;
    debug_mode = true;
    do_siftmatch = true;
    confidence_thresh = 0.8f;
    image_scale_for_sift = 0.5;
    best_knn = 9999;
  }
};

class MicroGPS {
public:
  MicroGPS();
  ~MicroGPS();

  void setVotingCellSize(float m_grid_step_);
  void setNumScaleGroups(int num_scale_groups);
  void loadDatabaseOnly(Dataset* dataset);

  void removeDuplicates();
  void computePCABasis();
  void PCAreduction(int num_dimensions_to_keep);
  void preprocessDatabaseImages (int num_samples_per_image, float image_scale_for_sift);

  // void preprocessDatabase(Database* database,
  //                         int n_samples_per_image,
  //                         int num_dimensions_to_keep);

  void buildSearchIndex();
  void searchNearestNeighbors(WorkImage* work_image, std::vector<int>& nn_index);

  void buildSearchIndexMultiScales();
  void searchNearestNeighborsMultiScales(WorkImage* work_image, std::vector<int>& nn_index, int best_k = 9999);

  void savePCABasis(const char* path);
  void loadPCABasis(const char* path);
  void saveFeatures(const char* path);
  void loadFeatures(const char* path);
  bool locate(WorkImage* work_image, WorkImage*& alignment_image,
              MicroGPSOptions& options,
              MicroGPSResult& result,
              MicroGPSTiming& timing, 
              MicroGPSDebug& debug);

private:
  Eigen::MatrixXf m_PCA_basis;  
  Eigen::MatrixXf m_features;
  Eigen::MatrixXf m_features_PCAed;
  std::vector<Eigen::Matrix3f> m_feature_poses;
  std::vector<int> m_feature_image_idx;
  std::vector<float> m_feature_scales;
  Eigen::MatrixXf m_feature_local_location;

  std::vector<WorkImage*> m_database_images;

  // FLANN data
  flann::Matrix<float> m_features_PCAed_flann;
  flann::Index<L2<float> >* m_flann_kdtree;

  std::vector<flann::Matrix<float> > m_features_PCAed_flann_multi_scales;
  std::vector<flann::Index<L2<float> >* > m_flann_kdtree_multi_scales;
  std::vector<float> m_bounds_multi_scales;  
  std::vector<std::vector<int> > m_global_index_multi_scales;

  Dataset* m_database;
  // int m_num_samples_per_image;
  // int m_num_dimensions_to_keep;
  int m_image_width;
  int m_image_height;


  std::vector<int> m_voting_grid;
  float m_grid_step;
  float m_grid_min_x;
  float m_grid_min_y;
  int m_grid_width;
  int m_grid_height;

  int m_num_scale_groups;

};



#endif
