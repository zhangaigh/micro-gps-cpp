#ifndef _IMAGE_DATASET_H_
#define _IMAGE_DATASET_H_

#include <vector>
#include <Eigen/Dense>


namespace MicroGPS {

class ImageDataset {
public:
  ImageDataset(const char* packed_data_path);
  ~ImageDataset();

  void                          loadDatabaseImages();
  void                          loadDefaultTestSequence();
  void                          loadTestSequenceByName(const char* filename);
  void                          setPrecomputedFeatureSuffix(const char* suffix);

  size_t                        getDatabaseSize() const;
  const char*                   getDatabaseImagePath(const unsigned idx) const;
  Eigen::Matrix3f               getDatabaseImagePose(const unsigned idx) const;
  void                          getDatabaseImagePrecomputedFeatures(const unsigned idx, 
                                                                    char* const path, 
                                                                    const char* suffix = NULL) const;

  size_t                        getTestSequenceSize() const;
  const char*                   getTestImagePath(const unsigned idx) const;
  void                          getTestImagePrecomputedFeatures(const unsigned idx, 
                                                                char* const path,
                                                                const char* suffix = NULL) const;

private:
  char                          m_packed_data_path[256];
  char                          m_test_sequence_name[256];
  char                          m_precomputed_feature_suffix[256];

  // database
  std::vector<Eigen::Matrix3f>  m_database_poses;
  std::vector<char*>            m_database_images;
  std::vector<char*>            m_test_images;
};

}


#endif
