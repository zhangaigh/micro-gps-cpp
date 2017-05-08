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
char* g_screenshots_root  = (char*)("screenshots");
char* g_test_results_root = (char*)("test_results");
#endif


// variables that can be changed by command line / GUI
char  g_dataset_name[256];
char  g_testset_name[256];
char  g_test_results_name[256];
char  g_feature_database_name[256];
char  g_pca_basis_name[256];
char  g_precomputed_feature_suffix[256];

char  g_map_name[256];
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

// basic variables 
int g_num_frames_tested = 0;
int g_num_frames_succeeded = 0;
int g_test_index = 0;

// variables used by GUI
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

void EventPreprocessing() {
  char selected_database_path[256];
  sprintf(selected_database_path, "%s/%s", g_database_root,
                                           g_feature_database_name);
  char selected_pca_basis_path[256];
  sprintf(selected_pca_basis_path, "%s/%s", g_PCA_basis_root,
                                            g_pca_basis_name);
  // compute precomputed values
  if (!util::checkFileExists(selected_database_path)) { // create if not exists
    g_localizer->preprocessDatabaseImages(g_database_sample_size, g_sift_extraction_scale);
    g_localizer->saveFeatures(selected_database_path);
  }

  // compute pca basis
  if (!util::checkFileExists(selected_pca_basis_path)) { // create if not exists
    g_localizer->computePCABasis();  
    char s[256]; 
    sprintf(s, "%s/pca_%s", g_PCA_basis_root, g_feature_database_name); // use standard name    
    g_localizer->savePCABasis(s);    
  }  
}

void EventInitLocalizer() {
  g_localizer->setVotingCellSize(g_cell_size);
  g_localizer->setNumScaleGroups(g_num_scale_groups);
  g_localizer->loadImageDataset(g_dataset);

  char selected_database_path[256];
  sprintf(selected_database_path, "%s/%s", g_database_root,
                                           g_feature_database_name);
  char selected_pca_basis_path[256];
  sprintf(selected_pca_basis_path, "%s/%s", g_PCA_basis_root,
                                            g_pca_basis_name);

  EventPreprocessing(); // run only if files don't exist
  // reload precomputed values
  g_localizer->loadFeatures(selected_database_path);
  // reload pca basis
  g_localizer->loadPCABasis(selected_pca_basis_path);
 
  g_localizer->dimensionReductionPCA(g_dimensionality);
  g_localizer->buildSearchIndexMultiScales();
}

void EventTestCurrentFrame() {
  char precomputed_feat_path[256];
  char precomputed_sift_path[256];

  g_dataset->getTestImagePrecomputedFeatures(0, precomputed_feat_path);
  g_dataset->getTestImagePrecomputedFeatures(1, precomputed_sift_path, (char*)("sift"));

  MicroGPS::Image* current_test_frame = 
        new MicroGPS::Image(g_dataset->getTestImagePath(g_test_index),
                            precomputed_feat_path,
                            precomputed_sift_path);

  current_test_frame->loadImage();

  MicroGPS::Image* alignment_image;
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
  delete alignment_image;
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
    }

    // load test sequence
    util::listDir(selected_dataset_path, g_testset_list, ".test", true); // list test sequences
    ImGui::Combo("###testset_name", &g_testset_selected_idx, g_testset_list);
    ImGui::SameLine();
    if (ImGui::Button("load test", ImVec2(-1, 0))) {
      g_dataset->loadTestSequenceByName(g_testset_list[g_testset_selected_idx].c_str());
    }

    // load map image with scale
    util::listDir(g_map_image_root, g_map_image_list, "", true); // list maps
    ImGui::Combo("map image###map image", &g_map_image_selected_idx, g_map_image_list);
    ImGui::Combo("map scale###map_scale", &g_map_scale_selected_idx, g_map_scale_list);
    // ImGui::SameLine();
    if (ImGui::Button("load map", ImVec2(-1, 0))) {
      int percentage;
      sscanf(g_map_scale_list[g_map_scale_selected_idx].c_str(), "%d", &percentage);
      // eventLoadMap(mgpsVars.load_map_image_path[mgpsVars.load_map_image_path_selected].c_str(), (double)percentage / 100.0f);
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

    if (ImGui::Button("reload", ImVec2(-1, 0))) {
      strcpy(g_feature_database_name, g_database_list[g_database_selected_idx].c_str());
      strcpy(g_pca_basis_name, g_pca_basis_list[g_pca_basis_selected_idx].c_str());
      EventInitLocalizer();
    }

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
      // eventTestAll(true);
    }
    ImGui::Checkbox("Debug",      &g_localizer_options.m_save_debug_info);
    ImGui::SameLine();
    ImGui::Checkbox("SIFT-Match", &g_localizer_options.m_do_siftmatch_verification);
    ImGui::SameLine();
    ImGui::Checkbox("Alignment",  &g_localizer_options.m_generate_alignment_image);

    // eventTestAll(false);
    
    // update visualization of the test frame
    static int prev_test_index = -1;
    if (g_test_index != prev_test_index) {
      // WorkImage* current_test_frame = new WorkImage(mgpsVars.dataset->getTestImage(g_test_index));
      // current_test_frame->loadImage();
      // mgpsVars.test_image_texture.loadTextureFromWorkImage(current_test_frame);
      // delete current_test_frame;
      prev_test_index = g_test_index;
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

    if (ImGui::Button("locate", ImVec2(-1, 0))) {
      EventTestCurrentFrame();
    }
  }

  // ======================================== Training ========================================
  ImGui::SetNextTreeNodeOpen(true, ImGuiSetCond_Once);
  if (ImGui::CollapsingHeader("Training")) {
    char save_map_image_name[256];
    ImGui::InputText("map path", save_map_image_name, 256);
    static int save_map_scale_selected_idx = 0;
    ImGui::Combo("map scale###save", &save_map_scale_selected_idx, g_map_scale_list);

    if (ImGui::Button("generate map", ImVec2(-1, 0))) {
      char s[256];
      sprintf(s, "%s/%s", g_map_image_root, save_map_image_name);
      int percentage;
      sscanf(g_map_scale_list[save_map_scale_selected_idx].c_str(), "%d", &percentage);
      // generateMapFromDataset(mgpsVars.dataset, s, (float)percentage / 100.0f);
    }
   
    ImGui::InputInt("sample size", &g_database_sample_size);
    if (g_database_sample_size <= 0) { // set some limit
      g_database_sample_size = 1;
    }

    char save_database_name[256];
    sprintf(save_database_name, "*-siftgpu.bin");
    char save_PCA_basis_name[256];
    sprintf(save_PCA_basis_name, "pca-*-siftgpu.bin");
    ImGui::InputText("database###save_database", save_database_name, 256);
    ImGui::InputText("PCA basis###save_pca_basis", save_PCA_basis_name, 256);

    if (ImGui::Button("process", ImVec2(-1, 0))) {
      strcpy(g_feature_database_name, g_database_list[g_database_selected_idx].c_str());
      strcpy(g_pca_basis_name, g_pca_basis_list[g_pca_basis_selected_idx].c_str());
      EventPreprocessing();
    }
  }

  // ======================================== Monitor ========================================
  ImGui::SetNextTreeNodeOpen(true, ImGuiSetCond_Once);
  if (ImGui::CollapsingHeader("Monitor")) {

    ImVec2 size = ImGui::GetContentRegionAvail();
   
    // eventPrintResults();

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
      // ImGui::PushFont(io.Fonts->Fonts[2]);
      drawSetting();
      // ImGui::PopFont();

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

  delete g_localizer;

  return 0;
}
