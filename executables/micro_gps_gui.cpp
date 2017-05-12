// C/C++
#include <stdio.h>
#include <fstream>
#include <vector>

// Dependencies
#include <imgui.h>
#include "imgui_impl_glfw_gl3.h"
#include <GL/gl3w.h>
#include <GLFW/glfw3.h>
#include <gflags/gflags.h>

// Project related
#include "micro_gps.h"
#include "util.h"
#include "gui_helper.h"


#define GUI_TEST_WIDTH    (g_glfw_display.screen_h/2 - GUI_GAP_SIZE/2)
#define GUI_SETTING_WIDTH 400
#define GUI_GAP_SIZE      10


#ifdef ON_MAC
char* g_dataset_root      = (char*)("/Users/lgzhang/Documents/DATA/micro_gps_packed");
char* g_map_image_root    = (char*)("maps");
char* g_database_root     = (char*)("databases");
char* g_PCA_basis_root    = (char*)("pca_bases");
char* g_visual_words_root = (char*)("visual_words");
// char* g_screenshots_root  = (char*)("screenshots");
char* g_test_results_root = (char*)("test_results");
#endif


// variables that can be changed by command line / GUI
char  g_dataset_name[256];
char  g_testset_name[256];
char  g_test_results_name[256];
char  g_feature_database_name[256];
char  g_pca_basis_name[256];
char  g_visual_words_name[256];
char  g_precomputed_feature_suffix[256];

char  g_map_image_name[256];
float g_map_scale;

float g_cell_size;
int   g_num_scale_groups;
int   g_dimensionality;
int   g_database_sample_size;
float g_sift_extraction_scale;


MicroGPS::Localization*        g_localizer = NULL;
MicroGPS::ImageDataset*        g_dataset = NULL;
MicroGPS::LocalizationOptions  g_localizer_options;
MicroGPS::LocalizationResult   g_localizer_result;
MicroGPS::LocalizationTiming   g_localizer_timing;
// MicroGPS::Image*               g_map_image;

// basic variables 
int g_num_frames_tested = 0;
int g_num_frames_succeeded = 0;
int g_prev_test_index = 0;
int g_test_index = 0;

// variables used by GUI
// TODO: I think *selected_idx should be local variables
GLFWDisplay               g_glfw_display;
std::vector<std::string>  g_dataset_list;
int                       g_dataset_selected_idx = 0;
std::vector<std::string>  g_testset_list;
int                       g_testset_selected_idx = 0;
std::vector<std::string>  g_map_image_list;
int                       g_map_image_selected_idx = 0;
std::vector<std::string>  g_map_scale_list = {"10%", "25%", "50%", "100%"};
int                       g_map_scale_selected_idx = 0;
std::vector<std::string>  g_database_list;
int                       g_database_selected_idx = 0;
std::vector<std::string>  g_pca_basis_list;
int                       g_pca_basis_selected_idx = 0;
std::vector<std::string>  g_visual_words_list;
int                       g_visual_words_selected_idx = 0;

float                     g_map_texture_avail_width;
float                     g_map_texture_avail_height;
float                     g_map_texture_screen_pos_x;
float                     g_map_texture_screen_pos_y;
float                     g_map_texture_display_scale;
float                     g_map_texture_display_w;
float                     g_map_texture_display_h;

ImageGL3Texture           g_map_texture;
ImageGL3Texture           g_map_feature_pose_overlay_texture;
ImageGL3Texture           g_map_image_pose_overlay_texture;
ImageGL3Texture           g_test_image_texture;
ImageGL3Texture           g_alignment_texture;

bool                      g_draw_camera = true;
float                     g_world_min_x = 0.0f;
float                     g_world_min_y = 0.0f;


DEFINE_bool   (batch_test,        false,                                              "do batch test");
DEFINE_string (dataset_root,      "/Users/lgzhang/Documents/DATA/micro_gps_packed",   "dataset_root");
DEFINE_string (dataset,           "fc_hallway_long_packed",                           "dataset to use");
DEFINE_string (testset,           "test00.test",                                      "test sequence");
DEFINE_string (output,            "tests",                                            "output");
DEFINE_string (feature_db,        "fc_hallway_long_packed-siftgpu.bin",               "database features");
DEFINE_string (pca_basis,         "pca_fc_hallway_long_packed-siftgpu.bin",           "pca basis to use");
DEFINE_string (map,               "fc_map_10per.png",                                 "stitched map");
DEFINE_double (map_scale,         0.1,                                                "map scale");
DEFINE_double (cell_size,         50.0f,                                              "size of the voting cell");
DEFINE_int32  (num_scale_groups,  10,                                                 "number of search indexes");
DEFINE_int32  (feat_dim,          8,                                                  "dimensionality after PCA reduction");
DEFINE_int32  (best_knn,          9999,                                               "use the best k nearest neighbors for voting");
DEFINE_double (sift_ext_scale,    0.5,                                                "extract sift at this scale");
DEFINE_bool   (test_all,          false,                                              "test all frames");
DEFINE_bool   (nogui,             false,                                              "disable gui");
// offline
DEFINE_int32  (db_sample_size,    50,                                                 "number of features sampled from each database image");
DEFINE_string (feat_suffix,       "sift",                                             "default suffix for precomputed feature");


void LoadVariablesFromCommandLine() {
  // TODO: overwrite g* variables with gflags values
  // g_dataset_root = (char*)FLAGS_dataset_root.c_str();
  strcpy(g_dataset_name,                  FLAGS_dataset.c_str());
  strcpy(g_testset_name,                  FLAGS_testset.c_str());
  strcpy(g_test_results_name,             FLAGS_output.c_str());
  strcpy(g_feature_database_name,         FLAGS_feature_db.c_str());
  strcpy(g_pca_basis_name,                FLAGS_pca_basis.c_str());
  strcpy(g_precomputed_feature_suffix,    FLAGS_feat_suffix.c_str());

  g_cell_size                           = FLAGS_cell_size;
  g_num_scale_groups                    = FLAGS_num_scale_groups;
  g_dimensionality                      = FLAGS_feat_dim;
  g_database_sample_size                = FLAGS_db_sample_size;
  g_sift_extraction_scale               = FLAGS_sift_ext_scale;

  printf("g_dataset_name=%s\n", g_dataset_name);
}


void computeMapOffsets() {
  int n_images = (int)g_dataset->getDatabaseSize();
  // int n_images = 100;
  std::vector<MicroGPS::Image*> work_images(n_images);
  std::vector<Eigen::Matrix3f> work_image_poses(n_images);

  for (int i = 0; i < n_images; i++) {
    work_images[i] = new MicroGPS::Image(g_dataset->getDatabaseImagePath(i));
    work_image_poses[i] = g_dataset->getDatabaseImagePose(i);
  }

  int world_min_x;
  int world_min_y;
  int world_max_x;
  int world_max_y;

  MicroGPS::ImageFunc::computeImageArrayWorldLimits(work_images,
                                                    work_image_poses,
                                                    g_map_scale,
                                                    world_min_x,
                                                    world_min_y,
                                                    world_max_x,
                                                    world_max_y);

  for (int i = 0; i < n_images; i++) {
    delete work_images[i];
  }

  g_world_min_x = (float)world_min_x;
  g_world_min_y = (float)world_min_y;
}

float globalLength2TextureLength(float x) {
  return x * g_map_scale * g_map_texture_display_scale;
}

void globalCoordinates2TextureCoordinates(float& x, float& y) {
  x = (x * g_map_scale - g_world_min_x) * g_map_texture_display_scale;
  y = (y * g_map_scale - g_world_min_y) * g_map_texture_display_scale;
}


void saveGUIRegion(int topleft_x, int topleft_y, int width, int height,
                   const char* out_path) {

  int multiplier_x = g_glfw_display.framebuffer_w / g_glfw_display.screen_w;
  int multiplier_y = g_glfw_display.framebuffer_h / g_glfw_display.screen_h;

  // we just read RGB
  MicroGPS::Image screenshot(width * multiplier_x, height * multiplier_y, 3);

  int lowerleft_x = topleft_x;
  int lowerleft_y = g_glfw_display.screen_h - (topleft_y + height);

  glReadBuffer(GL_FRONT); // wth is GL_FRONT_LEFT / GL_FRONT???
  glPixelStorei(GL_PACK_ALIGNMENT, 1); // fixing the "multiples of 4" problem
  glReadPixels(lowerleft_x * multiplier_x, lowerleft_y * multiplier_y,
   	          width * multiplier_x, height * multiplier_y,
             	GL_BGR,
             	GL_UNSIGNED_BYTE,
             	screenshot.data());

  screenshot.flip(0);

  char s[256];
  sprintf(s, "%s/%s/%s", g_test_results_root, g_test_results_name, out_path);
  screenshot.write(s);
}

MicroGPS::Image* generateDistributionMap(std::vector<Eigen::Vector2f> points, 
                                         float& vmin_out, float& vmax_out,
                                         float vmin_in = -1.0, float vmax_in = -1.0) {
  printf("generateDistributionMap\n");
  int w = round(g_map_texture_display_scale * g_map_texture.raw_width);
  int h = round(g_map_texture_display_scale * g_map_texture.raw_height);


  cv::Mat map = cv::Mat::zeros(h, w, CV_32FC1);
  int cnt = 0;
  for (size_t i = 0; i < points.size(); i++) {
    int x = (int)floor(points[i].x());
    int y = (int)floor(points[i].y());
    if (x < 0 || x > w-1 || y < 0 || y > h-1 ) {
      continue;
    }
    cnt ++;
    map.at<float>(y, x) += 1.0f;
  }

  cv::GaussianBlur(map, map, cv::Size(21, 21), 5.0f);

  double vmin, vmax;
  if (vmin_in >= 0.0 && vmax_in >= 0.0) {
    vmin = vmin_in;
    vmax = vmax_in;
    printf("vmax = %f, vmin = %f\n", vmax, vmin);
  } else {
    cv::minMaxLoc(map, &vmin, &vmax);
    vmin_out = vmin;
    vmax_out = vmax;
  }

  cv::Mat map_jet(h, w, CV_32FC3);
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      float r,g,b;
      float val = map.at<float>(y, x);
      MicroGPS::ImageFunc::gray2jet(val, vmin, vmax, r, g, b);
      map_jet.at<cv::Vec3f>(y, x)[0] = b;
      map_jet.at<cv::Vec3f>(y, x)[1] = g;
      map_jet.at<cv::Vec3f>(y, x)[2] = r;
    }
  }

  map_jet *= 255.0f;
  map_jet.convertTo(map_jet, CV_8UC3);

  MicroGPS::Image* map_jet_image = new MicroGPS::Image(w, h, 3);
  memcpy(map_jet_image->data(), map_jet.data, w*h*3);

  return map_jet_image;
}



void EventPreprocessing(bool create_anyway = true) {
  char selected_database_path[256];
  sprintf(selected_database_path, "%s/%s", g_database_root,
                                           g_feature_database_name);
  char selected_pca_basis_path[256];
  sprintf(selected_pca_basis_path, "%s/%s", g_PCA_basis_root,
                                            g_pca_basis_name);
  
  printf("selected_database_path = %s\n", selected_database_path);
  printf("selected_pca_basis_path = %s\n", selected_pca_basis_path);
  printf("start preprocessing\n");
  
  // compute precomputed values
  if (!util::checkFileExists(selected_database_path) || create_anyway) { // create if not exists
    g_localizer->preprocessDatabaseImages(g_database_sample_size, g_sift_extraction_scale);
    g_localizer->saveFeatures(selected_database_path);
  }

  // compute pca basis
  if (!util::checkFileExists(selected_pca_basis_path) || create_anyway) { // create if not exists
    g_localizer->computePCABasis();  
    char s[256]; 
    sprintf(s, "%s/pca-%s", g_PCA_basis_root, g_feature_database_name); // use standard name    
    g_localizer->savePCABasis(s);    
  }  
  printf("done\n");
}

void EventInitLocalizer() {
  g_localizer->setVotingCellSize(g_cell_size);
  g_localizer->setNumScaleGroups(g_num_scale_groups);

  char selected_database_path[256];
  sprintf(selected_database_path, "%s/%s", g_database_root,
                                           g_feature_database_name);
  char selected_pca_basis_path[256];
  sprintf(selected_pca_basis_path, "%s/%s", g_PCA_basis_root,
                                            g_pca_basis_name);

  EventPreprocessing(false); // run only if files don't exist
  // reload precomputed values
  g_localizer->loadFeatures(selected_database_path);
  // reload pca basis
  g_localizer->loadPCABasis(selected_pca_basis_path);
 
  g_localizer->dimensionReductionPCA(g_dimensionality);
  g_localizer->buildSearchIndexMultiScales();

  // prepare the result folder
  char test_results_path[256];
  sprintf(test_results_path, "%s/%s", g_test_results_root, g_test_results_name);
  util::mkdirIfNotExists(test_results_path);
}

void EventTestCurrentFrame() {
  char precomputed_feat_path[256];
  char precomputed_sift_path[256];

  g_dataset->getTestImagePrecomputedFeatures(g_test_index, precomputed_feat_path);
  g_dataset->getTestImagePrecomputedFeatures(g_test_index, precomputed_sift_path, (char*)("sift"));

  MicroGPS::Image* current_test_frame = 
        new MicroGPS::Image(g_dataset->getTestImagePath(g_test_index),
                            precomputed_feat_path,
                            precomputed_sift_path);

  current_test_frame->loadImage();

  MicroGPS::Image* alignment_image = NULL;
  g_localizer_timing.reset();
  g_localizer_result.reset();

  // TODO: set options
  // g_localizer_options.reset();
  g_localizer->locate(current_test_frame,
                      &g_localizer_options, 
                      &g_localizer_result,
                      &g_localizer_timing, 
                      alignment_image);

  current_test_frame->release();

  delete current_test_frame;

  if (alignment_image) {
    g_alignment_texture.loadTextureFromImage(alignment_image);
    delete alignment_image;
  } else {
    g_alignment_texture.deactivate();
  }

  if (g_localizer_options.m_save_debug_info) {
    int n_candidates = std::min(3000, (int)g_localizer_result.m_candidate_image_poses.size());
    std::vector<Eigen::Vector2f> image_origins(n_candidates);
    for (int fidx = 0; fidx < n_candidates; fidx++) {
      float kp_x = g_localizer_result.m_candidate_image_poses[fidx](0, 2);
      float kp_y = g_localizer_result.m_candidate_image_poses[fidx](1, 2);
      globalCoordinates2TextureCoordinates(kp_x, kp_y);
      image_origins[fidx](0) = kp_x;
      image_origins[fidx](1) = kp_y;
    }


    float vmin_out, vmax_out;
    MicroGPS::Image* image_pose_distribution = generateDistributionMap(image_origins,
                                                                      vmin_out, vmax_out);
    g_map_image_pose_overlay_texture.loadTextureFromImage(image_pose_distribution, g_map_texture.rotated90);
    delete image_pose_distribution;
    printf("vmax_out = %f, vmin_out = %f\n", vmax_out, vmin_out);

    int n_keypoints = std::min(3000, (int)g_localizer_result.m_matched_feature_poses.size());
    std::vector<Eigen::Vector2f> keypoints(n_keypoints);
    for (int fidx = 0; fidx < n_keypoints; fidx++) {
      float kp_x = g_localizer_result.m_matched_feature_poses[fidx](0, 2);
      float kp_y = g_localizer_result.m_matched_feature_poses[fidx](1, 2);
      globalCoordinates2TextureCoordinates(kp_x, kp_y);
      keypoints[fidx](0) = kp_x;
      keypoints[fidx](1) = kp_y;
    }

    float vmin_in = vmin_out;
    float vmax_in = vmax_out;
    MicroGPS::Image* feature_pose_distribution = generateDistributionMap(keypoints,
                                                                        vmin_out, vmax_out,
                                                                        vmin_in, vmax_in);
    g_map_feature_pose_overlay_texture.loadTextureFromImage(feature_pose_distribution, g_map_texture.rotated90);
    delete feature_pose_distribution;
  } else {
    g_map_image_pose_overlay_texture.deactivate();
    g_map_feature_pose_overlay_texture.deactivate();
  }
}

void EventGenerateMapFromDataset() {
  char s[256];
  sprintf(s, "%s/%s", g_map_image_root, g_map_image_name);

  size_t n_images = g_dataset->getDatabaseSize();
  // size_t n_images = 100;
  std::vector<MicroGPS::Image*> work_images(n_images);
  std::vector<Eigen::Matrix3f> work_image_poses(n_images);

  for (size_t i = 0; i < n_images; i++) {
    work_images[i] = new MicroGPS::Image(g_dataset->getDatabaseImagePath(i));
    work_image_poses[i] = g_dataset->getDatabaseImagePose(i);
  }

  printf("all work images loaded\n");
  MicroGPS::Image* map_image = MicroGPS::ImageFunc::warpImageArray(work_images, work_image_poses, g_map_scale);
  map_image->write(s);

  for (size_t i = 0; i < n_images; i++) {
    delete work_images[i];
  }
}

void EventLoadMap() {
  char s[256];
  sprintf(s, "%s/%s", g_map_image_root, g_map_image_name);

  if (!util::checkFileExists(s)) {
    EventGenerateMapFromDataset();
  }

  MicroGPS::Image* new_map = new MicroGPS::Image(s);
  new_map->loadImage();
  bool rotate90 = false;
  if (g_map_texture_avail_height > g_map_texture_avail_width != 
      new_map->height() > new_map->width()) {
    rotate90 = true;
  }
  g_map_texture.loadTextureFromImage(new_map, rotate90);
  delete new_map;

  computeMapOffsets();
}

void EventTestAll (bool trigger = false) {
  static bool to_save = false;
  static bool to_test = false;

  if (trigger) {
    to_save = false;
    to_test = true;    
    g_num_frames_tested = 0;
    g_num_frames_succeeded = 0;
    return;
  }

  static char s[256];
  // save screenshots after the previous frame is rendered
  if (to_save) {
    saveGUIRegion((int)g_map_texture_screen_pos_x+1, (int)g_map_texture_screen_pos_y,
                  (int)g_map_texture_display_w, (int)g_map_texture_display_h,
                  s);
    if (g_test_index < g_dataset->getTestSequenceSize() - 1) {
      g_test_index++;
      to_test = true;
    }
    // else if (FLAGS_batch_test) {
    //   exit(0);
    // }
    to_save = false;
  }

  if (to_test) {
    EventTestCurrentFrame();
    sprintf(s, "frame%06d.png", g_test_index);
    to_save = true;
    to_test = false;
    g_num_frames_tested++;
    g_num_frames_succeeded += (int)g_localizer_result.m_success_flag;
  } 
}


void EventPrintResults() {
  ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                                                              ImGui::GetIO().Framerate);


  ImGui::Text("Test result: %s (%d/%d) = %f%%", 
                                      g_localizer_result.m_success_flag ? "SUCCESS" : "FAILURE",
                                      g_num_frames_succeeded, g_num_frames_tested,
                                      (float)g_num_frames_succeeded / 
                                      (float)g_num_frames_tested * 100.0f);
  ImGui::Text("dx=%.02f, dy=%.02f da=%.02f\n", g_localizer_result.m_x_error, 
                                               g_localizer_result.m_y_error, 
                                               g_localizer_result.m_angle_error);
  ImGui::Text("Total time: %.3lf ms", g_localizer_timing.m_total);
  ImGui::BulletText("SIFT extraction : %.3lf ms", 
                                g_localizer_timing.m_sift_extraction);
  ImGui::BulletText("KNN search : %.3lf ms", 
                                g_localizer_timing.m_knn_search);
  ImGui::BulletText("Compute candidate poses : %.3lf ms", 
                                g_localizer_timing.m_candidate_image_pose);
  ImGui::BulletText("Voting : %.3lf ms", g_localizer_timing.m_voting);
  ImGui::BulletText("RANSAC : %.3lf ms", g_localizer_timing.m_ransac);

  if (g_localizer_result.m_top_cells.size() > 0) {
    int hist_max = *max_element(g_localizer_result.m_top_cells.begin(), 
                                g_localizer_result.m_top_cells.end());
    int n_cells = g_localizer_result.m_top_cells.size();
    float* data = new float[n_cells];
    for (int i = 0; i < n_cells; i++) {
      data[i] = (float)g_localizer_result.m_top_cells[i];
    }

    ImGui::Columns(n_cells, NULL, true);
    ImGui::Separator();
    for (int i = 0; i < n_cells; i++) {
        ImGui::Text("%.0f", data[i]);
        ImGui::NextColumn();
    }
    ImGui::Columns(1);
    ImGui::Separator();
  }
}

void EventLoadVisualWords() {
  char selected_visual_words_path[256];
  sprintf(selected_visual_words_path, "%s/%s", g_visual_words_root,
                                               g_visual_words_name);

  g_localizer->loadVisualWords(selected_visual_words_path);
  g_localizer->dimensionReductionPCAVisualWords();
  g_localizer->buildVisualWordsSearchIndex();
  g_localizer->fillVisualWordCells();
}


void drawSetting() {
  ImGui::SetNextWindowSize(ImVec2(GUI_SETTING_WIDTH, g_glfw_display.screen_h));
  ImGui::SetNextWindowPos(ImVec2(g_glfw_display.screen_w-GUI_SETTING_WIDTH, 0));

  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
  ImGui::Begin("Settings", NULL,  ImGuiWindowFlags_NoCollapse|
                                  ImGuiWindowFlags_NoResize|
                                  ImGuiWindowFlags_NoMove);

  // ======================================== Dataset ========================================
  ImGui::SetNextTreeNodeOpen(true, ImGuiSetCond_Once);
  if (ImGui::CollapsingHeader("Dataset")){

    // load dataset
    util::listDir(g_dataset_root, g_dataset_list, "", true); // list datasets
    ImGui::Combo("###dataset_name", &g_dataset_selected_idx, g_dataset_list);
    ImGui::SameLine();
    char selected_dataset_path[256];
    sprintf(selected_dataset_path, "%s/%s", g_dataset_root, 
                                            g_dataset_list[g_dataset_selected_idx].c_str());
    if (ImGui::Button("load dataset", ImVec2(-1, 0))) {
      if (g_dataset) {
        delete g_dataset;
      }
      g_dataset = new MicroGPS::ImageDataset(selected_dataset_path);
      g_dataset->loadDatabaseImages();
      g_localizer->loadImageDataset(g_dataset);
    }

    // load test sequence
    util::listDir(selected_dataset_path, g_testset_list, ".test", true); // list test sequences
    ImGui::Combo("###testset_name", &g_testset_selected_idx, g_testset_list);
    ImGui::SameLine();
    if (ImGui::Button("load test", ImVec2(-1, 0))) {
      g_dataset->loadTestSequenceByName(g_testset_list[g_testset_selected_idx].c_str());
      g_prev_test_index = -1; // trigger refreshing
      g_test_index = 0;
    }

    // load map image with scale
    util::listDir(g_map_image_root, g_map_image_list, "", true); // list maps
    ImGui::Combo("map image###map image", &g_map_image_selected_idx, g_map_image_list);
    ImGui::Combo("map scale###map_scale", &g_map_scale_selected_idx, g_map_scale_list);
    // ImGui::SameLine();
    if (ImGui::Button("load map", ImVec2(-1, 0))) {
      int percentage;
      sscanf(g_map_scale_list[g_map_scale_selected_idx].c_str(), "%d", &percentage);
      g_map_scale = (float)(percentage) / 100.0f;
      strcpy(g_map_image_name, g_map_image_list[g_map_image_selected_idx].c_str());
      EventLoadMap();
    }
  }

  // ======================================== Testing ========================================
  ImGui::SetNextTreeNodeOpen(true, ImGuiSetCond_Once);
  if (ImGui::CollapsingHeader("Testing") ) {
    
    ImGui::InputFloat("cell size", &g_cell_size, 1.0f, 0.0f, 1);
    if (g_cell_size <= 0) { // set some limit
      g_cell_size = 1.0f;
    }
    ImGui::InputInt("scale groups", &g_num_scale_groups);
    if (g_num_scale_groups <= 0) { // set some limit
      g_num_scale_groups = 1;
    }

    ImGui::InputInt("dimensionality", &g_dimensionality);
    if (g_dimensionality <= 0) { // set some limit
      g_dimensionality = 1;
    }

    // feature database
    util::listDir(g_database_root, g_database_list, "", true); // list databases
    ImGui::Combo("database", &g_database_selected_idx, g_database_list);

    // PCA basis
    util::listDir(g_PCA_basis_root, g_pca_basis_list, "", true); // list PCA bases
    ImGui::Combo("PCA basis", &g_pca_basis_selected_idx, g_pca_basis_list);

    // visual words
    util::listDir(g_visual_words_root, g_visual_words_list, "", true);
    ImGui::Combo("visual words", &g_visual_words_selected_idx, g_visual_words_list);
    

    if (ImGui::Button("reload", ImVec2(180, 0))) {
      strcpy(g_feature_database_name, g_database_list[g_database_selected_idx].c_str());
      strcpy(g_pca_basis_name, g_pca_basis_list[g_pca_basis_selected_idx].c_str());
      EventInitLocalizer();
    }
    ImGui::SameLine();
    if (ImGui::Button("load vw", ImVec2(180, 0))) {
      strcpy(g_visual_words_name, g_visual_words_list[g_visual_words_selected_idx].c_str());
      EventLoadVisualWords();
    }

    // if (ImGui::Button("load vw cells", ImVec2(-1, 0))) {
    //   g_localizer->loadVisualWordCells("vw_cells.bin");
    // }


    int max_test_index = 9999;
    if (g_dataset) {
      max_test_index = g_dataset->getTestSequenceSize()-1;
    }
    ImGui::SliderInt("image index", &g_test_index, 0, max_test_index);
    // ImGui::PushItemWidth(30);
    // ImGui::SameLine();
    if (ImGui::Button("-", ImVec2(30, 0))) {
      g_test_index--;
    }
    ImGui::SameLine();
    if (ImGui::Button("+", ImVec2(30, 0))) {
      g_test_index++;
    }
    ImGui::SameLine();
    static char test_all_prefix[256] = "prefix";
    ImGui::PushItemWidth(100);
    ImGui::InputText("###test_all_prefix", test_all_prefix, 256);
    ImGui::PopItemWidth();
    ImGui::SameLine();
    if (ImGui::Button("test all")) {
      EventTestAll(true);
    }
    ImGui::Checkbox("Debug",      &g_localizer_options.m_save_debug_info);
    ImGui::SameLine();
    ImGui::Checkbox("SIFT-Match", &g_localizer_options.m_do_siftmatch_verification);
    ImGui::SameLine();
    ImGui::Checkbox("Alignment",  &g_localizer_options.m_generate_alignment_image);

    EventTestAll(false);
    
    // update visualization of the test frame
    if (g_test_index != g_prev_test_index && 
        g_dataset != NULL &&
        g_dataset->getTestSequenceSize() > 0) {
      MicroGPS::Image* current_test_frame = 
                      new MicroGPS::Image(g_dataset->getTestImagePath(g_test_index));
      current_test_frame->loadImage();
      g_test_image_texture.loadTextureFromImage(current_test_frame);
      delete current_test_frame;
      g_prev_test_index = g_test_index;
    }

    ImGui::InputFloat("sift scale", &g_localizer_options.m_image_scale_for_sift, 0.05f, 0.0f, 2);
    if (g_localizer_options.m_image_scale_for_sift <= 0.0) { // set some limit
      g_localizer_options.m_image_scale_for_sift = 0.05;
    } else if (g_localizer_options.m_image_scale_for_sift > 1.0) {
      g_localizer_options.m_image_scale_for_sift = 1.0;      
    }

    ImGui::InputInt("best kNN", &g_localizer_options.m_best_knn);
    if (g_localizer_options.m_best_knn <= 0) { // set some limit
      g_localizer_options.m_best_knn = 1;
    }

    static int locate_method = 0;
    ImGui::RadioButton("NN", &locate_method, 0); ImGui::SameLine();
    ImGui::RadioButton("VW", &locate_method, 1); ImGui::SameLine();

    if (locate_method) {
      g_localizer_options.m_use_visual_words = true;
    } else {
      g_localizer_options.m_use_visual_words = false;      
    }

    if (ImGui::Button("locate", ImVec2(-1, 0))) {
      EventTestCurrentFrame();
    }
  }

  // ======================================== Training ========================================
  ImGui::SetNextTreeNodeOpen(false, ImGuiSetCond_Once);
  if (ImGui::CollapsingHeader("Training")) {
    static char save_map_image_name[256] = "map.png";
    // sprintf(save_map_image_name, "map.png");
    ImGui::InputText("map path", save_map_image_name, 256);
    static int save_map_scale_selected_idx = 0;
    ImGui::Combo("map scale###save", &save_map_scale_selected_idx, g_map_scale_list);

    if (ImGui::Button("generate map", ImVec2(-1, 0))) {
      int percentage;
      sscanf(g_map_scale_list[save_map_scale_selected_idx].c_str(), "%d", &percentage);
      g_map_scale = (float)percentage / 100.0f;
      strcpy(g_map_image_name, save_map_image_name);      
      EventGenerateMapFromDataset();
    }
   
    ImGui::InputInt("sample size", &g_database_sample_size);
    if (g_database_sample_size <= 0) { // set some limit
      g_database_sample_size = 1;
    }

    static char save_database_name[256] = "*-siftgpu.bin";
    // sprintf(save_database_name, "*-siftgpu.bin");
    static char save_PCA_basis_name[256] = "pca-*-siftgpu.bin";
    // sprintf(save_PCA_basis_name, "pca-*-siftgpu.bin");
    ImGui::InputText("database###save_database", save_database_name, 256);
    ImGui::InputText("PCA basis###save_pca_basis", save_PCA_basis_name, 256);

    if (ImGui::Button("process", ImVec2(-1, 0))) {
      strcpy(g_feature_database_name, save_database_name);
      strcpy(g_pca_basis_name, save_PCA_basis_name);
      EventPreprocessing();
    }

    static char save_vw_cells_name[256] = "vw_cells.bin";
    ImGui::InputText("###save_vw_cells", save_vw_cells_name, 256);
    ImGui::SameLine();
    if (ImGui::Button("vw cells", ImVec2(-1, 0))) {
      g_localizer->fillVisualWordCells();
      g_localizer->saveVisualWordCells(save_vw_cells_name);
    }
  }

  // ======================================== Monitor ========================================
  ImGui::SetNextTreeNodeOpen(true, ImGuiSetCond_Once);
  if (ImGui::CollapsingHeader("Monitor")) {

    ImVec2 size = ImGui::GetContentRegionAvail();
   
    EventPrintResults();

    static bool show_test_window = false; 
    if (ImGui::Button("demo")) {
      show_test_window ^= 1;
    }

    if (show_test_window) {
      ImGui::SetNextWindowPos(ImVec2(650, 20), ImGuiSetCond_FirstUseEver);
      ImGui::ShowTestWindow(&show_test_window);
    }

    static bool show_style_editor = false; 
    if (ImGui::Button("style")) {
      show_style_editor ^= 1;
    }

    if (show_style_editor) {
      ImGui::SetNextWindowPos(ImVec2(650, 20), ImGuiSetCond_FirstUseEver);
      ImGui::ShowStyleEditor();
    }


  }

  ImGui::End();
  ImGui::PopStyleVar();

}



void drawMapViewer() {
  ImGui::SetNextWindowSize(ImVec2(g_glfw_display.screen_w-GUI_TEST_WIDTH-GUI_SETTING_WIDTH-GUI_GAP_SIZE*2,
                                  g_glfw_display.screen_h));
  ImGui::SetNextWindowPos(ImVec2(0, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
  ImGui::Begin("Map", NULL,  ImGuiWindowFlags_NoCollapse|
                             ImGuiWindowFlags_NoResize  |
                             ImGuiWindowFlags_NoMove    |
                             ImGuiWindowFlags_NoScrollbar);

  ImVec2 region_avail = ImGui::GetContentRegionAvail(); // excluding padding
  region_avail.y -= ImGui::GetItemsLineHeightWithSpacing() * 2;  // reserving space for other widgets

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(2.0f, 2.0f));
  ImGui::BeginChild("map_drawing_region", ImVec2(region_avail.x, region_avail.y), true);

  ImVec2 im_disp_region = ImGui::GetContentRegionAvail();
  g_map_texture_avail_width = im_disp_region.x;
  g_map_texture_avail_height = im_disp_region.y;;

  static int overlay_transparency = 128;
  static int overlay_idx = 1;

  // display the image
  if (g_map_texture.active) {
    // compute texture size
    g_map_texture_display_scale = std::min(g_map_texture_avail_width /           
                                           g_map_texture.width,
                                           g_map_texture_avail_height / 
                                           g_map_texture.height);
    float tex_w = g_map_texture.width * g_map_texture_display_scale;
    float tex_h = g_map_texture.height * g_map_texture_display_scale;
    
    if (g_map_texture_avail_width > tex_w) {
      ImGui::Indent((g_map_texture_avail_width - tex_w) / 2.0f);
    }
    ImVec2 tex_screen_pos = ImGui::GetCursorScreenPos(); // save cursor pose

    // update texture info
    g_map_texture_display_w = tex_w;
    g_map_texture_display_h = tex_h;
    g_map_texture_screen_pos_x = tex_screen_pos.x;
    g_map_texture_screen_pos_y = tex_screen_pos.y;


    ImGui::Image((void*)g_map_texture.id, ImVec2(tex_w, tex_h), 
                  ImVec2(0,0), ImVec2(1,1), 
                  ImColor(255,255,255,255), ImColor(0,0,0,0));
    
    ImGui::SetCursorScreenPos(tex_screen_pos); // go back


    switch (overlay_idx) {
      case 0:
        if (g_map_feature_pose_overlay_texture.active) {
          ImGui::Image((void*)g_map_feature_pose_overlay_texture.id, ImVec2(tex_w, tex_h), 
                        ImVec2(0,0), ImVec2(1,1), 
                        ImColor(255,255,255,overlay_transparency), ImColor(0,0,0,0));
        }
        break;
      case 1:
        if (g_map_image_pose_overlay_texture.active) {
          ImGui::Image((void*)g_map_image_pose_overlay_texture.id, ImVec2(tex_w, tex_h), 
                        ImVec2(0,0), ImVec2(1,1), 
                        ImColor(255,255,255,overlay_transparency), ImColor(0,0,0,0));
        }
        break;
    }

    // draw estimated location / orientation
    if (g_localizer_result.m_can_estimate_pose) {
      float center_x, center_y;
      float x_axis_x, x_axis_y;
      float y_axis_x, y_axis_y;
      
      center_x = g_localizer_result.m_final_estimated_pose(0, 2);
      center_y = g_localizer_result.m_final_estimated_pose(1, 2);
      globalCoordinates2TextureCoordinates(center_x, center_y);

      if (g_map_texture.rotated90) {
        float tmp = center_x;
        center_x = g_map_texture_display_w - center_y;
        center_y = tmp;
        x_axis_y = g_localizer_result.m_final_estimated_pose(0, 0);
        x_axis_x = -g_localizer_result.m_final_estimated_pose(1, 0);
        y_axis_y = g_localizer_result.m_final_estimated_pose(0, 1);
        y_axis_x = -g_localizer_result.m_final_estimated_pose(1, 1);        
      } else {
        x_axis_x = g_localizer_result.m_final_estimated_pose(0, 0);
        x_axis_y = g_localizer_result.m_final_estimated_pose(1, 0);
        y_axis_x = g_localizer_result.m_final_estimated_pose(0, 1);
        y_axis_y = g_localizer_result.m_final_estimated_pose(1, 1);
      }

      center_x += tex_screen_pos.x;
      center_y += tex_screen_pos.y;

      if (g_draw_camera) {
        ImGui::GetWindowDrawList()->AddLine(ImVec2(center_x, center_y), 
                                            ImVec2(center_x + x_axis_x*20, center_y + x_axis_y*20),
                                            ImColor(0,0,255,255), 4.0f);
        ImGui::GetWindowDrawList()->AddLine(ImVec2(center_x, center_y), 
                                            ImVec2(center_x + y_axis_x*20, center_y + y_axis_y*20),
                                            ImColor(255,0,0,255), 4.0f);

        ImColor circle_color;
        if (g_localizer_result.m_success_flag) {
          circle_color = ImColor(0, 255, 0, 255);
        } else {
          circle_color = ImColor(255, 0, 0, 255);          
        }
        ImGui::GetWindowDrawList()->AddCircle(ImVec2(center_x, center_y), 20, 
                                              circle_color, 24, 4.0f);

        // float frame_width = globalLength2TextureLength(1288);
        // float frame_height = globalLength2TextureLength(964);
        // ImGui::GetWindowDrawList()->AddLine(ImVec2(center_x, center_y), ImVec2(center_x + x_axis_x*frame_width, center_y + x_axis_y*frame_width),
        //                                       ImColor(255,0,0,255), 2.0f);
        // ImGui::GetWindowDrawList()->AddLine(ImVec2(center_x, center_y), ImVec2(center_x + y_axis_x*frame_height, center_y + y_axis_y*frame_height),
        //                                       ImColor(0,0,255,255), 2.0f);
        // ImGui::GetWindowDrawList()->AddLine(ImVec2(center_x + x_axis_x*frame_width + y_axis_x*frame_height, center_y + y_axis_y*frame_height + x_axis_y*frame_width), ImVec2(center_x + x_axis_x*frame_width, center_y + x_axis_y*frame_width),
        //                                       ImColor(255,0,0,255), 2.0f);
        // ImGui::GetWindowDrawList()->AddLine(ImVec2(center_x + x_axis_x*frame_width + y_axis_x*frame_height, center_y + y_axis_y*frame_height + x_axis_y*frame_width), ImVec2(center_x + y_axis_x*frame_height, center_y + y_axis_y*frame_height),
        //                                       ImColor(0,0,255,255), 2.0f);

      }
    }
  }

  ImGui::EndChild();
  ImGui::PopStyleVar();


  static char save_rendered_map_path[256];
  ImGui::PushItemWidth(250);
  ImGui::InputText("###save_rendered_map_path", save_rendered_map_path, 256);
  ImGui::PopItemWidth();
  ImGui::SameLine();

  if (ImGui::Button("save")) {
    printf("saving map screenshot: %s\n", save_rendered_map_path);
    saveGUIRegion((int)g_map_texture_screen_pos_x+1, (int)g_map_texture_screen_pos_y,
                  (int)g_map_texture_display_w, (int)g_map_texture_display_h,
                  save_rendered_map_path);
  }


  ImGui::PushItemWidth(150);
  ImGui::DragInt("overlay alpha", &overlay_transparency, 1.0f, 0, 255);
  ImGui::PopItemWidth();

  ImGui::SameLine();
  ImGui::RadioButton("NN pose", &overlay_idx, 0); ImGui::SameLine();
  ImGui::RadioButton("image pose", &overlay_idx, 1);
  
  ImGui::SameLine();
  ImGui::Checkbox("Camera", &g_draw_camera);


  ImGui::End();
  ImGui::PopStyleVar();

  
}


void drawTestImageViewer() {
  ImGui::SetNextWindowSize(ImVec2(GUI_TEST_WIDTH, GUI_TEST_WIDTH));
  ImGui::SetNextWindowPos(ImVec2(g_glfw_display.screen_w-GUI_TEST_WIDTH-GUI_SETTING_WIDTH-GUI_GAP_SIZE, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
  ImGui::Begin("Test Image", NULL,  ImGuiWindowFlags_NoCollapse |
                                    ImGuiWindowFlags_NoResize   |
                                    ImGuiWindowFlags_NoMove     |
                                    ImGuiWindowFlags_NoScrollbar);


  ImVec2 region_avail = ImGui::GetContentRegionAvail(); // excluding padding
  region_avail.y -= ImGui::GetItemsLineHeightWithSpacing();  // reserving space for other widgets

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(2.0f, 2.0f));
  ImGui::BeginChild("test_image_disp", ImVec2(region_avail.x, region_avail.y), true);

  ImVec2 tex_screen_pos;
  float tex_w;
  float tex_h;

  // display the image
  if (g_test_image_texture.active) {
    // compute texture size
    ImVec2 im_disp_region = ImGui::GetContentRegionAvail();
    float im_scaling = std::min(im_disp_region.x / g_test_image_texture.width, 
                                im_disp_region.y / g_test_image_texture.height);
    tex_w = g_test_image_texture.width * im_scaling;
    tex_h = g_test_image_texture.height * im_scaling;

    if (im_disp_region.x > tex_w) {
      ImGui::Indent((im_disp_region.x - tex_w) / 2.0f);
    }

    tex_screen_pos = ImGui::GetCursorScreenPos();


    ImGui::Image((void*)g_test_image_texture.id, 
                  ImVec2(tex_w, tex_h), ImVec2(0,0), ImVec2(1,1), 
                  ImColor(255,255,255,255), ImColor(0,0,0,0));

    // ImGui::GetWindowDrawList()->AddImage((void*)g_test_image_texture.id, 
    //                                       ImVec2(tex_screen_pos.x, tex_screen_pos.y),
    //                                       ImVec2(tex_screen_pos.x+tex_w, tex_screen_pos.y+tex_h));

  }

  ImGui::EndChild();
  ImGui::PopStyleVar();


  static char save_rendered_test_image_path[256];
  ImGui::PushItemWidth(250);
  ImGui::InputText("###save_rendered_test_image_path", save_rendered_test_image_path, 256);
  ImGui::PopItemWidth();
  ImGui::SameLine();

  if (ImGui::Button("save")) {
    // printf("save image\n");
    saveGUIRegion((int)tex_screen_pos.x+1, (int)tex_screen_pos.y, 
                  (int)tex_w, (int)tex_h,
                  save_rendered_test_image_path);
  }

  ImGui::End();
  ImGui::PopStyleVar();
}

void drawAlignmentImageViewer() {
  ImGui::SetNextWindowSize(ImVec2(GUI_TEST_WIDTH, GUI_TEST_WIDTH));
  ImGui::SetNextWindowPos(ImVec2(g_glfw_display.screen_w-GUI_TEST_WIDTH-GUI_SETTING_WIDTH-GUI_GAP_SIZE, GUI_TEST_WIDTH+GUI_GAP_SIZE));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
  ImGui::Begin("Alignment", NULL, ImGuiWindowFlags_NoCollapse |
                                  ImGuiWindowFlags_NoResize   |
                                  ImGuiWindowFlags_NoMove     |
                                  ImGuiWindowFlags_NoScrollbar);

  ImVec2 region_avail = ImGui::GetContentRegionAvail(); // excluding padding
  region_avail.y -= ImGui::GetItemsLineHeightWithSpacing();  // reserving space for other widgets

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(2.0f, 2.0f));
  ImGui::BeginChild("test_image_disp", ImVec2(region_avail.x, region_avail.y), true);


  ImVec2 tex_screen_pos;
  float tex_w;
  float tex_h;

  // display the image
  if (g_alignment_texture.active) {
    // compute texture size
    ImVec2 im_disp_region = ImGui::GetContentRegionAvail();
    float im_scaling = std::min(im_disp_region.x / g_alignment_texture.width, 
                                im_disp_region.y / g_alignment_texture.height);
    tex_w = g_alignment_texture.width * im_scaling;
    tex_h = g_alignment_texture.height * im_scaling;

    if (im_disp_region.x > tex_w) {
      ImGui::Indent((im_disp_region.x - tex_w) / 2.0f);
    }

    tex_screen_pos = ImGui::GetCursorScreenPos();
    ImGui::Image((void*)g_alignment_texture.id, 
                  ImVec2(tex_w, tex_h), ImVec2(0,0), ImVec2(1,1), 
                  ImColor(255,255,255,255), ImColor(0,0,0,0));
  }

  ImGui::EndChild();
  ImGui::PopStyleVar();

  static char save_rendered_alignment_path[256];
  ImGui::PushItemWidth(250);
  ImGui::InputText("###save_rendered_alignment_path", save_rendered_alignment_path, 256);
  ImGui::PopItemWidth();
  ImGui::SameLine();

  if (ImGui::Button("save")) {
    saveGUIRegion((int)tex_screen_pos.x+1, (int)tex_screen_pos.y, 
                  (int)tex_w, (int)tex_h,
                  save_rendered_alignment_path);
  }


  ImGui::SameLine();

  ImGui::End();
  ImGui::PopStyleVar();
}



int main(int argc, char *argv[]) {
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  printf("Arguments parsed\n");
  
  LoadVariablesFromCommandLine();
  // readRobotCameraCalibration(robot_camera_calibration_file_path);

  MicroGPS::initSiftGPU();
  g_localizer = new MicroGPS::Localization();
  
  // Setup window
  glfwSetErrorCallback(gui_error_callback);
  if (!glfwInit())
      return 1;
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
  GLFWwindow* window = glfwCreateWindow(1440, 900, "MicroGPS OpenGL3 GUI", NULL, NULL);
  glfwMakeContextCurrent(window);
  gl3wInit();

  // Setup ImGui binding
  ImGui_ImplGlfwGL3_Init(window, true);

  // Load Fonts
  // (there is a default font, this is only if you want to change it. see extra_fonts/README.txt for more details)
  ImGuiIO& io = ImGui::GetIO();
  // io.Fonts->AddFontDefault();
  io.Fonts->AddFontFromFileTTF("../imgui/lib/extra_fonts/Cousine-Regular.ttf", 15.0f);
  // io.Fonts->AddFontFromFileTTF("../imgui/lib/extra_fonts/DroidSans.ttf", 15.0f);
  // io.Fonts->AddFontFromFileTTF("../imgui/lib/extra_fonts/ProggyClean.ttf", 15.0f);
  // io.Fonts->AddFontFromFileTTF("../imgui/lib/extra_fonts/ProggyTiny.ttf", 10.0f);
  // io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, NULL, 
  //                              io.Fonts->GetGlyphRangesJapanese());

  
  ImVec4 clear_color = ImColor(114, 144, 154);

  printf("entering glfw loop\n");

  // Main loop
  while (!glfwWindowShouldClose(window))
  {
      glfwPollEvents();
      ImGui_ImplGlfwGL3_NewFrame();

      glfwGetWindowSize(window, &g_glfw_display.screen_w, &g_glfw_display.screen_h);
      glfwGetFramebufferSize(window, &g_glfw_display.framebuffer_w, &g_glfw_display.framebuffer_h);

      // drawGui();
      drawSetting();
      drawMapViewer();
      drawTestImageViewer();
      drawAlignmentImageViewer();

      // Rendering
      glViewport(0, 0, g_glfw_display.screen_w, g_glfw_display.screen_h);
      glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      ImGui::Render();
      glfwSwapBuffers(window);
  }


  // Cleanup
  ImGui_ImplGlfwGL3_Shutdown();
  glfwTerminate();

  if (g_localizer){
    delete g_localizer;
  }
  if (g_dataset) {
    delete g_dataset;
  }
  

  return 0;
}
