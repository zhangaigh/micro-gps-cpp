#ifndef _GUI_H_
#define _GUI_H_

#include "micro_gps.h"

struct WorkImageGL3Texture {
  GLuint id;
  float width; // original texture width (no matter rotated or not)
  float height;
  bool show;
  bool rotated90;

  void loadTextureFromWorkImage(WorkImage* work_image, bool rotate90 = false);
  void loadTextureFromCVMat(cv::Mat& image, bool rotate90 = false);
  void disable();
  float getWidth(); // get (rotated) width
  float getHeight(); 

  WorkImageGL3Texture();
};


struct GLFWDisplay {
  int screen_w;
  int screen_h;
  int framebuffer_w;
  int framebuffer_h;
};

struct RenderedTextureInfo {
  float screen_pos_x;
  float screen_pos_y;
  float fitting_scale;
  float width;
  float height;
  bool rotated90;
};


struct MicroGPSVariables {
  // Roots
  char dataset_root[256];
  char map_image_root[256];
  char database_root[256];
  char PCA_basis_root[256];
  char screenshots_root[256];
  char test_results_root[256];

  // Dataset
  std::vector<std::string> dataset_path;
  int dataset_path_selected;
  std::vector<std::string> load_map_image_path;
  int load_map_image_path_selected;
  std::vector<std::string> load_test_sequence_path;
  int load_test_sequence_path_selected;
  std::vector<std::string> map_scales;
  int load_map_scale_selected;

  // Testing
  float cell_size;
  int scale_groups;
  std::vector<std::string> load_database_path;
  int load_database_path_selected;
  std::vector<std::string> load_PCA_basis_path;
  int load_PCA_basis_path_selected;
  int prev_test_index;
  int test_index;
  bool to_test_current_frame;
  bool to_save_tested_frame;

  // Training
  int feature_sample_size;
  int PCA_dimensions;
  char save_map_image_path[256];
  char save_database_path[256];
  char save_PCA_basis_path[256];


  // Monitor
  std::vector<float> top_cells_histogram;
  bool success_flag;
  int num_frames_tested;
  int num_frames_succeeded;

  // Visualization
  WorkImageGL3Texture map_texture;
  WorkImageGL3Texture map_feature_pose_overlay_texture;
  WorkImageGL3Texture map_image_pose_overlay_texture;
  WorkImageGL3Texture test_image_texture;
  WorkImageGL3Texture alignment_texture;

  RenderedTextureInfo map_texture_info;

  float map_texture_avail_width;
  float map_texture_avail_height;


  bool show_test_window;

  bool draw_camera;

  // Micro-GPS internals
  Database* dataset;
  MicroGPS* micro_gps;
  MicroGPSOptions options;
  MicroGPSResult result;
  MicroGPSTiming timing;
  MicroGPSDebug debug;

  float world_min_x;
  float world_min_y;

  // bool enable_alignment;

  void loadDefaultValues();
};







#endif
