#include <fstream>
#include <vector>
#include <gflags/gflags.h>
#include <stdio.h>

#include "micro_gps.h"
#include "util.h"


#include <imgui.h>
#include "imgui_impl_glfw_gl3.h"
#include <GL/gl3w.h>
#include <GLFW/glfw3.h>

// // OpenGL related
// #include "imgui.h"
// #include "imgui_impl_glfw_gl3.h"
// #include <GL/gl3w.h>
// #include <GLFW/glfw3.h>
// #include "gui.h"

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


MicroGPS*       g_micro_gps = NULL;
Dataset*        g_dataset = NULL;
MicroGPSOptions g_micro_gps_options;
MicroGPSResult  g_micro_gps_result;
MicroGPSTiming  g_micro_gps_timing;
MicroGPSDebug   g_micro_gps_debug;

// basic variables 
int g_num_frames_tested = 0;
int g_num_frames_succeeded = 0;
int g_test_index = 0;

// void InitWithDefaultVariables() {
//   g_variable_cell_size = 50.0f;
//   g_variable_num_scale_groups = 10;
//   g_dimensionality = 8;
// }



DEFINE_bool   (batch_test,        false,                                              "do batch test");
DEFINE_string (dataset_root,      "/Users/lgzhang/Documents/DATA/micro_gps_packed",   "dataset_root");
DEFINE_string (dataset,           "fc_hallway_long_packed",                         "dataset to use");
DEFINE_string (testset,           "test00.test",                                      "test sequence");
DEFINE_string (output,            "tests",                                            "output");
DEFINE_string (feature_db,        "fc_hallway_long_packed-siftgpu.bin",               "database features");
DEFINE_string (pca_basis,         "pca_fc_hallway_long_packed-siftgpu.bin",           "pca basis to use");
DEFINE_string (map,               "acee_asphalt_map_10per.png",                       "stitched map");
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
// precomputed_feature
// DEFINE_string (test_precomputed, "", "test images precomputed features - .txt");
// DEFINE_string (dataset_precomputed, "", "database images precomputed features - .txt");


void LoadVariablesFromCommandLine() {
  // TODO: overwrite g* variables with gflags values
  // strcpy(g_database_root,         FLAGS_dataset_root.c_str());
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
}


void commandLineBatchTest() {
  if (g_dataset != NULL) { // reset if not NULL
    delete g_dataset;
  }

  char dataset_path[256];
  sprintf(dataset_path, "%s/%s", g_dataset_root, g_dataset_name);

  g_dataset = new Dataset(dataset_path);
  g_dataset->loadDatabase();
  g_dataset->loadTestSequenceByName(g_testset_name);
  g_dataset->setPrecomputedFeatureSuffix(g_precomputed_feature_suffix);

  if (g_micro_gps != NULL) {
    delete g_micro_gps;
  }

  g_micro_gps = new MicroGPS();
  g_micro_gps->loadDatabaseOnly(g_dataset);

  printf("Loaded dataset\n");

  g_micro_gps->setVotingCellSize(g_cell_size);
  g_micro_gps->setNumScaleGroups(g_num_scale_groups);
  g_micro_gps->loadDatabaseOnly(g_dataset);
  printf("Micro-GPS configured\n");

  // reload precomputed values
  char s[256];
  sprintf(s, "%s/%s", g_database_root, g_feature_database_name);
  if (!checkFileExists(s)) { // create if not exists
    g_micro_gps->preprocessDatabaseImages(g_database_sample_size, g_sift_extraction_scale);
    g_micro_gps->saveFeatures(s);
    printf("Feature database computed\n");
  }
  g_micro_gps->loadFeatures(s);
  printf("Feature database loaded\n");
  
  if (strcmp(g_pca_basis_name, "") != 0) {
    sprintf(s, "%s/%s", g_PCA_basis_root, g_pca_basis_name);
    g_micro_gps->loadPCABasis(s);
  } else {
    g_micro_gps->computePCABasis();
    sprintf(s, "%s/pca_%s", g_PCA_basis_root, g_feature_database_name);
    g_micro_gps->savePCABasis(s);
    printf("PCA basis computed\n");
  }
  printf("PCA basis loaded\n");

  g_micro_gps->PCAreduction(g_dimensionality);
  printf("Reduced feature dimensionality\n");

  g_micro_gps->buildSearchIndexMultiScales();
  printf("Built search index\n");


  char test_report_folder[256];
  sprintf(test_report_folder, "%s/%s", g_test_results_root, g_test_results_name);
  mkdirIfNotExists(test_report_folder);

  for (int test_index = 0; test_index < g_dataset->getTestSize(); test_index++) {
    bool success_flag = false;
    WorkImage* current_test_frame = new WorkImage(g_dataset->getTestImage(test_index),
                                                  g_dataset->getTestPrecomputedFeatures(test_index));
    current_test_frame->loadImage();

    WorkImage* alignment_image = NULL;
    g_micro_gps_timing.reset();
    g_micro_gps_result.reset();
    g_micro_gps_debug.reset();  

    success_flag = g_micro_gps->locate(current_test_frame, alignment_image,
                                                        g_micro_gps_options, g_micro_gps_result,
                                                        g_micro_gps_timing, g_micro_gps_debug);
    g_micro_gps_result.success_flag = success_flag;
    current_test_frame->release();
    delete current_test_frame;
    delete alignment_image;

    char test_report_path[256];
    sprintf(test_report_path, "%s/frame%06d.txt", test_report_folder, test_index);
    FILE* fp = fopen(test_report_path, "w");
    g_micro_gps_timing.printToFile(fp);
    g_micro_gps_result.printToFile(fp);
    g_micro_gps_debug.printToFile(fp);
    fclose(fp);
  }
}


int main(int argc, char *argv[]) {
  initSiftGPU();


  gflags::ParseCommandLineFlags(&argc, &argv, true);
  printf("Arguments parsed\n");
  
  
  if (FLAGS_batch_test) {
    LoadVariablesFromCommandLine();
    printf("Arguments assigned\n");
  } else {
    // TODO: set variables from other sources
  }

  if (FLAGS_nogui) {
    commandLineBatchTest();
  }

}