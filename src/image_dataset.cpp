#include "stdio.h"
#include <string>
#include <iostream>


#include "image_dataset.h"
#include "util.h"

// #define RUN_TEST


namespace MicroGPS {


ImageDataset::ImageDataset(const char* packed_data_path) {
  strcpy(m_packed_data_path,            packed_data_path);
  strcpy(m_test_sequence_name,          (char*)(""));
  strcpy(m_precomputed_feature_suffix,  (char*)("sift"));
}


ImageDataset::~ImageDataset() {
  for (size_t i = 0; i < m_database_images.size(); i++) {
    delete[] m_database_images[i];
  }
  m_database_images.clear();

  for (size_t i = 0; i < m_test_images.size(); i++) {
    delete[] m_test_images[i];
  }
  m_test_images.clear();

}

void ImageDataset::loadDatabaseImages() {
  m_database_poses.clear();
  char p[256];

  sprintf(p, "%s/database.txt", m_packed_data_path);
  printf("Loading database images: %s\n", p);

  for (int i = 0; i < m_database_images.size(); i++) {
    delete[] m_database_images[i];
  }
  m_database_images.clear();

  char line[256];
  FILE* fp = fopen(p, "r");
  while (fgets(line, sizeof(line), fp) != NULL) {
    char tmp_path[256];
    sscanf(line, "%s\n", tmp_path);

    char* im_path = new char[256];
    sprintf(im_path, "%s/%s", m_packed_data_path, tmp_path);
    m_database_images.push_back(im_path);
    // printf("%s\n", im_path);

    fgets(line, sizeof(line), fp);
    Eigen::Matrix3f pose;
    char* data = line;
    for (int i = 0; i < 9; i++) {
      float val;
      int offset;
      sscanf(data, "%f%n", &val, &offset);
      pose(i/3, i%3) = val;
      data = data + offset;
    }

    m_database_poses.push_back(pose);

    // std::cout << pose << std::endl;
  }

  fclose(fp);
}


void ImageDataset::loadDefaultTestSequence() {
  this->loadTestSequenceByName("test.test");
}


void ImageDataset::loadTestSequenceByName(const char* filename) {
  strcpy(m_test_sequence_name, filename);
  
  for (int i = 0; i < m_test_images.size(); i++) {
    delete[] m_test_images[i];
  }
  m_test_images.clear();

  char p[256];
  sprintf(p, "%s/%s", m_packed_data_path, filename);
  printf("Loading test sequence: %s\n", p);

  char line[256];
  FILE* fp = fopen(p, "r");
  while (fgets(line, sizeof(line), fp) != NULL) {
    char tmp_path[256];
    sscanf(line, "%s\n", tmp_path);

    char* im_path = new char[256];
    sprintf(im_path, "%s/%s", m_packed_data_path, tmp_path);
    m_test_images.push_back(im_path);
    // printf("%s\n", im_path);
  }

  fclose(fp);

}

void ImageDataset::setPrecomputedFeatureSuffix(const char* suffix) {
  strcpy(m_precomputed_feature_suffix, suffix);
}

size_t ImageDataset::getDatabaseSize() const {
  return m_database_images.size();
}

const char* ImageDataset::getDatabaseImagePath(const unsigned idx) const {
  return (const char*)m_database_images[idx];
}

Eigen::Matrix3f ImageDataset::getDatabaseImagePose(const unsigned idx) const {
  return m_database_poses[idx];
}

void ImageDataset::getDatabaseImagePrecomputedFeatures(const unsigned idx, 
                                                      char* const path, 
                                                      const char* suffix) const
{
  char* suffix_used;
  if (suffix) {
    suffix_used = (char*)suffix;
  } else {
    suffix_used = (char*)m_precomputed_feature_suffix;
  }

  sprintf(path, "%s/precomputed_features/database.%s/frame%06d.bin", 
                  m_packed_data_path,
                  suffix_used,
                  idx);

  if (!checkFileExists(path)) {
    sprintf(path, "");
  }
}


size_t ImageDataset::getTestSequenceSize() const {
  return m_test_images.size();
}

const char* ImageDataset::getTestImagePath(const unsigned idx) const {
  return (const char*)m_test_images[idx];
}

void ImageDataset::getTestImagePrecomputedFeatures(const unsigned idx, 
                                                    char* const path,
                                                    const char* suffix) const
{
  char* suffix_used;
  if (suffix) {
    suffix_used = (char*)suffix;
  } else {
    suffix_used = (char*)m_precomputed_feature_suffix;
  }

  sprintf(path, "%s/precomputed_features/%s.%s/frame%06d.bin", 
                  m_packed_data_path,
                  m_test_sequence_name,
                  suffix_used,
                  idx);

  if (!checkFileExists(path)) {
    sprintf(path, "");
  }
}

}

#ifdef RUN_TEST

void test_print2_static_strings(const char* s1, const char* s2) {
  printf("s1 = %s\ns2 = %s\n", s1, s2);
}


int main(int argc, char const *argv[]) {
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

  test_print2_static_strings(s1, s2);
}

#endif

