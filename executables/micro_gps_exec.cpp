#include <fstream>
#include <vector>
#include <gflags/gflags.h>
#include <stdio.h>

#include "micro_gps.h"
#include "util.h"


// #include "imgui.h"
// #include "imgui_impl_glfw_gl3.h"
// #include <GL/gl3w.h>
// #include <GLFW/glfw3.h>


// #ifdef ON_MAC
// char* g_dataset_root      = (char*)("/Users/lgzhang/Documents/DATA/micro_gps_packed");
// #endif

// #ifdef ON_VISIONGPU1
// char* g_dataset_root      = (char*)("/data/linguang/micro_gps_packed");
// #endif

char  g_dataset_root[256];
char  g_feature_root[256];

char* g_database_root     = (char*)("databases");
char* g_PCA_basis_root    = (char*)("pca_bases");
char* g_visual_words_root = (char*)("visual_words");
char* g_test_results_root = (char*)("test_results");

// variables that can be changed by command line / GUI

// path
char  g_dataset_name[256];
char  g_testset_name[256];
char  g_test_results_name[256];
char  g_feature_database_name[256];
char  g_pca_basis_name[256];
char  g_visual_words_name[256];
char  g_precomputed_feature_suffix[256];

//  parameters
float g_cell_size;
int   g_num_scale_groups;
int   g_dimensionality;
int   g_best_knn;
int   g_database_sample_size;
float g_sift_extraction_scale;
int   g_frames_to_test;


MicroGPS::Localization*        g_localizer = NULL;
MicroGPS::ImageDataset*        g_dataset = NULL;
MicroGPS::LocalizationOptions  g_localizer_options;
MicroGPS::LocalizationResult   g_localizer_result;
MicroGPS::LocalizationTiming   g_localizer_timing;

// basic variables 
int g_num_frames_tested = 0;
int g_num_frames_succeeded = 0;
int g_test_index = 0;


// DEFINE_bool   (batch_test,        false,                                              "do batch test");
DEFINE_string (dataset_root,      "/Users/lgzhang/Documents/DATA/micro_gps_packed",   "dataset_root");
DEFINE_string (feature_root,      "/Users/lgzhang/Documents/DATA/micro_gps_packed_features", "feature root");
DEFINE_string (dataset,           "fc_hallway_long_packed",                           "dataset to use");
DEFINE_string (testset,           "test00.test",                                      "test sequence");
DEFINE_string (output,            "tests",                                            "output");
DEFINE_string (feature_db,        "tiles50-siftgpu.bin",                              "database features");
DEFINE_string (pca_basis,         "pca_tiles50-siftgpu.bin",                          "pca basis to use");
DEFINE_string (vw,                "",                                                 "visual words to use");

DEFINE_double (cell_size,         50.0f,                                              "size of the voting cell");
DEFINE_int32  (num_scale_groups,  10,                                                 "number of search indexes");
DEFINE_int32  (feat_dim,          8,                                                  "dimensionality after PCA reduction");
DEFINE_int32  (best_knn,          9999,                                               "use the best k nearest neighbors for voting");
DEFINE_double (sift_ext_scale,    0.5,                                                "extract sift at this scale");
DEFINE_int32  (frames_to_test,    9999999,                                            "max number of frames to test");
// offline
DEFINE_int32  (db_sample_size,    50,                                                 "number of features sampled from each database image");
DEFINE_string (feat_suffix,       "sift",                                             "default suffix for precomputed feature");
DEFINE_bool   (use_top_n,         false,                                              "use top n features with highest response to build database");

void LoadVariablesFromCommandLine() {
  // TODO: overwrite g* variables with gflags values
  strcpy(g_dataset_root,                  FLAGS_dataset_root.c_str());
  strcpy(g_feature_root,                  FLAGS_feature_root.c_str());
  strcpy(g_dataset_name,                  FLAGS_dataset.c_str());
  strcpy(g_testset_name,                  FLAGS_testset.c_str());
  strcpy(g_test_results_name,             FLAGS_output.c_str());
  strcpy(g_feature_database_name,         FLAGS_feature_db.c_str());
  strcpy(g_pca_basis_name,                FLAGS_pca_basis.c_str());
  strcpy(g_visual_words_name,             FLAGS_vw.c_str());
  strcpy(g_precomputed_feature_suffix,    FLAGS_feat_suffix.c_str());

  g_cell_size                           = FLAGS_cell_size;
  g_num_scale_groups                    = FLAGS_num_scale_groups;
  g_dimensionality                      = FLAGS_feat_dim;
  g_best_knn                            = FLAGS_best_knn;
  g_database_sample_size                = FLAGS_db_sample_size;
  g_sift_extraction_scale               = FLAGS_sift_ext_scale;
  g_frames_to_test                      = FLAGS_frames_to_test;

  printf("g_dataset_name=%s\n", g_dataset_name);
}


void commandLineBatchTest() {
  if (g_dataset != NULL) { // reset if not NULL
    delete g_dataset;
  }

  char dataset_path[256];
  sprintf(dataset_path, "%s/%s", g_dataset_root, g_dataset_name);
  char precomputed_feature_path[256];
  sprintf(precomputed_feature_path, "%s/%s", g_feature_root, g_dataset_name);

  g_dataset = new MicroGPS::ImageDataset(dataset_path, precomputed_feature_path);
  g_dataset->loadDatabaseImages();
  g_dataset->loadTestSequenceByName(g_testset_name);
  g_dataset->setPrecomputedFeatureSuffix(g_precomputed_feature_suffix);

  if (g_localizer != NULL) {
    delete g_localizer;
  }

  g_localizer = new MicroGPS::Localization();

  g_localizer->setVotingCellSize(g_cell_size);
  g_localizer->setNumScaleGroups(g_num_scale_groups);
  g_localizer->loadImageDataset(g_dataset);
  printf("Loaded dataset\n");
  printf("Micro-GPS configured\n");

  // reload precomputed values
  char s[256];
  sprintf(s, "%s/%s", g_database_root, g_feature_database_name);
  if (!util::checkFileExists(s)) { // create if not exists
    g_localizer->preprocessDatabaseImages(g_database_sample_size, g_sift_extraction_scale, FLAGS_use_top_n);
    g_localizer->saveFeatures(s);
    printf("Feature database computed\n");
  } else {
    g_localizer->loadFeatures(s);
    printf("Feature database loaded\n");
  }
  
  // load PCA basis
  sprintf(s, "%s/%s", g_PCA_basis_root, g_pca_basis_name);

  if (util::checkFileExists(s)) {
    g_localizer->loadPCABasis(s);
    printf("PCA basis loaded\n");
  } else {
    g_localizer->computePCABasis();
    sprintf(s, "%s/pca_%s", g_PCA_basis_root, g_feature_database_name);
    g_localizer->savePCABasis(s);
    printf("PCA basis computed\n");
  }


  g_localizer->dimensionReductionPCA(g_dimensionality);
  printf("Reduced feature dimensionality\n");

  g_localizer->buildSearchIndexMultiScales();
  printf("Built search index\n");

  // load visual words
  if (strcmp(g_visual_words_name, "")) {
    char selected_visual_words_path[256];
    sprintf(selected_visual_words_path, "%s/%s", g_visual_words_root,
                                                 g_visual_words_name);

    printf("Using visual words %s\n", selected_visual_words_path);
    g_localizer->loadVisualWords(selected_visual_words_path);
    g_localizer->dimensionReductionPCAVisualWords();
    g_localizer->buildVisualWordsSearchIndex();
    g_localizer->fillVisualWordCells();
  }


  char test_report_folder[256];
  sprintf(test_report_folder, "%s/%s", g_test_results_root, g_test_results_name);
  util::mkdirIfNotExists(test_report_folder);

  // set test options
  // struct LocalizationOptions {
  //   bool  m_save_debug_info;
  //   bool  m_generate_alignment_image;
  //   bool  m_do_match_verification;
  //   float m_image_scale_for_sift;
  //   int   m_best_knn;
  //   float m_confidence_thresh;  // not used
  //   bool  m_use_visual_words;
  // }
  
  g_localizer_options.m_save_debug_info           = true;
  g_localizer_options.m_generate_alignment_image  = false;
  g_localizer_options.m_do_match_verification     = true;
  g_localizer_options.m_image_scale_for_sift      = g_sift_extraction_scale;
  g_localizer_options.m_best_knn                  = g_best_knn;
  g_localizer_options.m_use_visual_words          = strcmp(g_visual_words_name, "");


  // for (int test_index = 0; test_index < g_dataset->getTestSequenceSize(); test_index++) {
  g_frames_to_test = std::min(g_frames_to_test, (int)g_dataset->getTestSequenceSize());
  
  int num_successful_frames = 0;

  for (int test_index = 0; test_index < g_frames_to_test; test_index++) {
    char precomputed_feat_path[256];
    char precomputed_sift_path[256];

    g_dataset->getTestImagePrecomputedFeatures(test_index, precomputed_feat_path);
    g_dataset->getTestImagePrecomputedFeatures(
      test_index, 
      precomputed_sift_path, 
      (char*)"key_siftgpu_desc_siftgpu_reso_0.5"
    );

    MicroGPS::Image* current_test_frame = 
          new MicroGPS::Image(g_dataset->getTestImagePath(test_index),
                              precomputed_feat_path,
                              precomputed_sift_path);

    // MicroGPS::Image* current_test_frame = 
    //       new MicroGPS::Image(g_dataset->getTestImagePath(test_index));

    current_test_frame->loadImage();

    MicroGPS::Image* alignment_image = NULL;
    g_localizer_timing.reset();
    g_localizer_result.reset();

    g_localizer->locate(current_test_frame,
                        &g_localizer_options, 
                        &g_localizer_result,
                        &g_localizer_timing, 
                        alignment_image);

    current_test_frame->release();

    delete current_test_frame;
    if (alignment_image) {
      delete alignment_image;
    }

    char test_report_path[256];
    printf("test_report_path = %s\n", test_report_path);

    sprintf(test_report_path, "%s/frame%06d.txt", test_report_folder, test_index);


    FILE* fp = fopen(test_report_path, "w");
    g_localizer_timing.printToFile(fp);
    g_localizer_result.printToFile(fp);
    fclose(fp);

    if (g_localizer_result.m_success_flag) {
      num_successful_frames++;
    }
  }
  printf("success rate: %d/%d = %f%\n", num_successful_frames, g_frames_to_test, float(num_successful_frames) / float(g_frames_to_test) * 100.0f);

}


int main(int argc, char *argv[]) {
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  printf("Arguments parsed\n");
  
  LoadVariablesFromCommandLine();

  printf("dataset_root = %s\n", g_dataset_root);

  MicroGPS::initSiftGPU();
  
  // if (FLAGS_batch_test) {
  //   LoadVariablesFromCommandLine();
  //   printf("Arguments assigned\n");
  // } else {
  //   // TODO: set variables from other sources
  // }

  commandLineBatchTest();

}
