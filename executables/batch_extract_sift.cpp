#include "micro_gps.h"
#include <sys/stat.h>
#include <gflags/gflags.h>


#if ON_MAC
  DEFINE_string(dataset_root, "/Users/lgzhang/Documents/DATA/micro_gps_packed", "dataset_root");
#endif
#if ON_AMD64
  DEFINE_string(dataset_root, "/data/linguang/micro_gps_packed", "dataset_root");
#endif

DEFINE_string(dataset, "nan", "dataset to use");
DEFINE_string(testset, "nan", "test sequence");
DEFINE_double(sift_extraction_scale, 0.5, "extract sift at this scale");
DEFINE_string(output, "features/nan", "where to output the features");
DEFINE_int32(rand_samples, 9999, "random sample features");
DEFINE_double(margin_thresh, -1.0f, "ignore points close to the boundaries");


void mkdirIfNotExists(const char* path) {
  struct stat buffer;   
  if (stat (path, &buffer) != 0) {
    printf("%s doesn't exist!\n", path);
    char cmd[256];
    sprintf(cmd, "mkdir %s", path);
    system(cmd);
  } else {
    printf("%s exists, ignore...\n", path);
  }
}


int main (int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

#ifdef USE_SIFT_GPU
  initSiftGPU();
#endif

  mkdirIfNotExists(FLAGS_output.c_str());
  char dataset_path[256];
  sprintf(dataset_path, "%s/%s", FLAGS_dataset_root.c_str(), FLAGS_dataset.c_str());

  bool extracting_test = false;
  Database* dataset = new Database(dataset_path);
  if (FLAGS_testset == "") {
    dataset->loadDatabase();
    extracting_test = false;
  } else {
    dataset->loadTestSequenceByName(FLAGS_testset.c_str());
    extracting_test = true;
  }

  int num_frames = extracting_test ? dataset->getTestSize() : dataset->getDatabaseSize();

  for (int i = 0; i < num_frames; i++ ){
    WorkImage* current_test_frame;
    if (extracting_test) {
      current_test_frame = new WorkImage(dataset->getTestImage(i));
    } else {
      current_test_frame = new WorkImage(dataset->getDatabaseImage(i));      
    }

    current_test_frame->loadImage();
    current_test_frame->extractSIFT(FLAGS_sift_extraction_scale);
    char s[256];
    sprintf(s, "%s/frame%06d.bin", FLAGS_output.c_str(), i);
    current_test_frame->saveSIFTFeatures(s, FLAGS_rand_samples, FLAGS_margin_thresh);
    delete current_test_frame;
  }

  delete dataset;
}