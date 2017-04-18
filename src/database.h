#ifndef _DATABASE_H_
#define _DATABASE_H_

#include <vector>
#include <Eigen/Dense>

class Database {
public:
  Database(const char* packed_data_path);
  ~Database();

  void loadDatabase();
  // void loadTestData();
  void loadDefaultTestSequence();
  void loadTestSequenceByName(const char*);

  void loadDatabasePrecomputedFeatures(const char*);
  void loadTestSequencePrecomputedFeatures(const char*);

  size_t getDatabaseSize();
  char* getDatabaseImage(int idx);
  Eigen::Matrix3f getDatabasePose(int idx);
  char* getDatabasePrecomputedFeatures(int idx);

  size_t getTestSize();
  char* getTestImage(int idx);
  char* getTestPrecomputedFeatures(int idx);

private:
  // const char* m_packed_data_path;
  char m_packed_data_path[256];

  // database
  std::vector<Eigen::Matrix3f> m_database_poses;
  std::vector<char*> m_database_images;
  std::vector<char*> m_database_precomputed_features;
  // test
  std::vector<char*> m_test_images;
  std::vector<char*> m_test_precomputed_features;
};


inline size_t Database::getDatabaseSize(){
  return m_database_images.size();
}
inline char* Database::getDatabaseImage(int idx){
  return m_database_images[idx];
}
inline char* Database::getDatabasePrecomputedFeatures(int idx) {
  if (m_database_precomputed_features.size() <= 0) {
    return NULL;
  }
  return m_database_precomputed_features[idx];
}

inline Eigen::Matrix3f Database::getDatabasePose(int idx){
  return m_database_poses[idx];
}

inline size_t Database::getTestSize(){
  return m_test_images.size();
}
inline char* Database::getTestImage(int idx){
  return m_test_images[idx];
}
inline char* Database::getTestPrecomputedFeatures(int idx) {
  if (m_test_precomputed_features.size() <= 0) {
    return NULL;
  }
  return m_test_precomputed_features[idx];
}



#endif
