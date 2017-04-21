// #include "image_dataset.h"
// #include "image.h"
#include "micro_gps2.h"


int main(int argc, char const *argv[]) {
  MicroGPS::initSiftGPU();

  MicroGPS::ImageDataset image_dataset("/Users/lgzhang/Documents/DATA/micro_gps_packed/fc_hallway_long_packed");

  image_dataset.loadDatabaseImages();
  // image_dataset.loadDefaultTestSequence();
  image_dataset.loadTestSequenceByName((const char*)("test00.test"));

  printf("database_images[0] = %s\n", image_dataset.getDatabaseImagePath(0));

  printf("test_images[0] = %s\n", image_dataset.getTestImagePath(0));


  char s1[256];
  char s2[256];

  image_dataset.getTestImagePrecomputedFeatures(0, s1);
  image_dataset.getTestImagePrecomputedFeatures(1, s2);

  MicroGPS::Image im(image_dataset.getTestImagePath(0),
                      s1,
                      s1);

  im.loadPrecomputedFeatures(true);

  printf("# local features = %ld\n", im.getNumLocalFeatures());


  MicroGPS::Localization localizer;

  localizer.loadImageDataset(&image_dataset);

  localizer.preprocessDatabaseImages(50, 0.5);

}