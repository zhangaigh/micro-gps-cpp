#ifndef _DATASET_H_
#define _DATASET_H_

#include <vector>
#include <Eigen/Dense>

class Dataset {
public:
  Dataset(const char* packed_data_path);
  ~Dataset();

  void loadDatabase();
  // void loadTestData();
  void loadDefaultTestSequence();
  void loadTestSequenceByName(const char*);

  void setPrecomputedFeatureSuffix(const char* suffix);

  size_t getDatabaseSize();
  char* getDatabaseImage(int idx);
  Eigen::Matrix3f getDatabasePose(int idx);
  char* getDatabasePrecomputedFeatures(int idx, const char* suffix = NULL);

  size_t getTestSize();
  char* getTestImage(int idx);
  char* getTestPrecomputedFeatures(int idx, const char* suffix = NULL);

private:
  // const char* m_packed_data_path;
  char m_packed_data_path[256];
  char m_test_sequence_name[256];
  char m_precomputed_feature_suffix[256];

  // database
  std::vector<Eigen::Matrix3f> m_database_poses;
  std::vector<char*> m_database_images;
  // std::vector<char*> m_database_precomputed_features;
  
  // test
  std::vector<char*> m_test_images;
  // std::vector<char*> m_test_precomputed_features;
};



#endif
