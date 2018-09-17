#include <gflags/gflags.h>
#include <sys/stat.h>
#include "micro_gps.h"

#if ON_MAC
DEFINE_string(dataset_root, "/Users/lgzhang/Documents/DATA/micro_gps_packed",
              "dataset root");
#endif

#if ON_LINUX
DEFINE_string(dataset_root, "/data/linguang/micro_gps_packed", "dataset root");
#endif

DEFINE_string(dataset, "", "dataset to use");
DEFINE_string(testset, "", "test sequence");
DEFINE_double(sift_ext_scale, 0.5, "extract sift at this scale");
DEFINE_string(output, "", "where to output the features");
DEFINE_int32(rand_samples, 9999, "random sample features");
DEFINE_double(margin_thresh, -1.0f, "ignore points close to the boundaries");

int main(int argc, char** argv) {
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

  MicroGPS::initSiftGPU();

  util::mkdirIfNotExists(FLAGS_output.c_str());
  char dataset_path[256];
  sprintf(dataset_path, "%s/%s", FLAGS_dataset_root.c_str(),
          FLAGS_dataset.c_str());

  bool extracting_test = false;
  MicroGPS::ImageDataset* dataset =
      new MicroGPS::ImageDataset(dataset_path, "");
  if (FLAGS_testset == "") {
    dataset->loadDatabaseImages();
    extracting_test = false;
  } else {
    dataset->loadTestSequenceByName(FLAGS_testset.c_str());
    extracting_test = true;
  }

  int num_frames = extracting_test ? dataset->getTestSequenceSize()
                                   : dataset->getDatabaseSize();

  for (int i = 0; i < num_frames; i++) {
    MicroGPS::Image* current_test_frame;
    if (extracting_test) {
      current_test_frame = new MicroGPS::Image(dataset->getTestImagePath(i));
    } else {
      current_test_frame =
          new MicroGPS::Image(dataset->getDatabaseImagePath(i));
    }

    current_test_frame->loadImage();
    current_test_frame->extractSIFT(FLAGS_sift_ext_scale);
    char s[256];
    sprintf(s, "%s/frame%06d.bin", FLAGS_output.c_str(), i);
    current_test_frame->saveLocalFeatures(s, FLAGS_rand_samples,
                                          FLAGS_margin_thresh);
    delete current_test_frame;
  }

  delete dataset;
}