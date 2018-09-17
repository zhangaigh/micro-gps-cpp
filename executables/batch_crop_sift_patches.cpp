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
// DEFINE_int32 (rand_samples,    9999,   "random sample features");
DEFINE_int32(patch_size, 64, "size of the patch to be cropped");
DEFINE_double(margin_thresh, -1.0f, "ignore points close to the boundaries");
DEFINE_string(feat_suffix, "sift", "default suffix for precomputed feature");

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

  dataset->setPrecomputedFeatureSuffix(FLAGS_feat_suffix.c_str());

  float margin_thresh = (float)FLAGS_margin_thresh;

  int num_frames = extracting_test ? dataset->getTestSequenceSize()
                                   : dataset->getDatabaseSize();

  for (int i = 0; i < num_frames; i++) {
    char precomputed_feat_path[256];
    char precomputed_sift_path[256];
    MicroGPS::Image* current_test_frame;

    if (extracting_test) {
      dataset->getTestImagePrecomputedFeatures(i, precomputed_feat_path);
      dataset->getTestImagePrecomputedFeatures(i, precomputed_sift_path,
                                               (char*)"sift");
      current_test_frame =
          new MicroGPS::Image(dataset->getTestImagePath(i),
                              precomputed_feat_path, precomputed_sift_path);
    } else {
      dataset->getDatabaseImagePrecomputedFeatures(i, precomputed_feat_path);
      dataset->getDatabaseImagePrecomputedFeatures(i, precomputed_sift_path,
                                                   (char*)"sift");
      current_test_frame =
          new MicroGPS::Image(dataset->getDatabaseImagePath(i),
                              precomputed_feat_path, precomputed_sift_path);
    }

    printf("test image path = %s\n", current_test_frame->getImagePath());

    current_test_frame->loadImage();

    // we only need gray image
    if (current_test_frame->channels() == 3) {
      current_test_frame->bgr2gray();
    }

    float w = (float)current_test_frame->width();
    float h = (float)current_test_frame->height();

    char s[256];
    sprintf(s, "%s/patches%06d.bin", FLAGS_output.c_str(), i);
    FILE* fp = fopen(s, "w");

    current_test_frame->loadPrecomputedFeatures(false);

    size_t patch_cnt = 0;
    for (size_t f_idx = 0; f_idx < current_test_frame->getNumLocalFeatures();
         f_idx++) {
      MicroGPS::LocalFeature* f = current_test_frame->getLocalFeature(f_idx);
      if (f->x > margin_thresh && f->x < w - 1 - margin_thresh &&
          f->y > margin_thresh && f->y < h - 1 - margin_thresh) {
        // crop
        MicroGPS::Image* patch = MicroGPS::ImageFunc::cropPatch(
            current_test_frame, f->x, f->y, f->angle, FLAGS_patch_size,
            FLAGS_patch_size);

        fwrite(patch->data(), sizeof(uchar), patch->width() * patch->height(),
               fp);
        patch_cnt++;

        delete patch;
      }
    }

    printf("cropped %lu / %lu patches\n", patch_cnt,
           current_test_frame->getNumLocalFeatures());

    fclose(fp);

    delete current_test_frame;
  }

  delete dataset;
}

// int main (int argc, char** argv) {
//   MicroGPS::Image lena("lena.bmp");
//   lena.loadImage();
//   printf("lena size = %ld x %ld x %ld\n", lena.width(), lena.height(),
//   lena.channels()); lena.bgr2gray();

//   // crop
//   MicroGPS::Image* patch = MicroGPS::ImageFunc::cropPatch(&lena, 256.0f,
//   256.0f, 45.0f / 180.0f * 3.14f,
//                                                           64, 64);

//   printf("patch size = %ld x %ld x %ld\n", patch->width(), patch->height(),
//   patch->channels());

//   patch->write("lena_cropped.png");

//   delete patch;

// }