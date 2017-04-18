#include "stdio.h"
#include <string>
#include <iostream>


#include "database.h"




Database::Database(const char* packed_data_path)
// m_packed_data_path(packed_data_path)
{
  strcpy(m_packed_data_path, packed_data_path);
}


Database::~Database() {
  for (size_t i = 0; i < m_database_images.size(); i++) {
    delete[] m_database_images[i];
  }
  m_database_images.clear();

  for (size_t i = 0; i < m_test_images.size(); i++) {
    delete[] m_test_images[i];
  }
  m_test_images.clear();

}

void Database::loadDatabase() {
  m_database_poses.clear();
  char p[256];

  sprintf(p, "%s/database.txt", m_packed_data_path);
  printf("%s\n", p);

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


void Database::loadDatabasePrecomputedFeatures(const char* filename) {
  m_database_precomputed_features.clear();
  char p[256];

  sprintf(p, "%s", filename);
  printf("%s\n", p);

  char line[256];
  FILE* fp = fopen(p, "r");
  while (fgets(line, sizeof(line), fp) != NULL) {
    char* path = new char[256];
    sscanf(line, "%s\n", path);
    m_database_precomputed_features.push_back(path);
    // printf("%s\n", im_path);
  }

  fclose(fp);
}


void Database::loadDefaultTestSequence() {
  this->loadTestSequenceByName("test.txt");
}


void Database::loadTestSequenceByName(const char* filename) {
  for (int i = 0; i < m_test_images.size(); i++) {
    delete[] m_test_images[i];
  }
  m_test_images.clear();

  char p[256];

  sprintf(p, "%s/%s", m_packed_data_path, filename);
  printf("%s\n", p);

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

void Database::loadTestSequencePrecomputedFeatures(const char* filename) {
  m_test_precomputed_features.clear();
  char p[256];

  // sprintf(p, "%s/%s", m_packed_data_path, filename);
  sprintf(p, "%s", filename);

  printf("%s\n", p);

  char line[256];
  FILE* fp = fopen(p, "r");
  while (fgets(line, sizeof(line), fp) != NULL) {
    char* path = new char[256];
    sscanf(line, "%s\n", path);
    m_test_precomputed_features.push_back(path);
    // printf("%s\n", im_path);
  }

  fclose(fp);
}

// int main(int argc, char const *argv[]) {
//   std::string s = "/Users/lgzhang/Documents/DATA/micro_gps_packed/fc_hallway_long_packed";
//   Database* database = new Database(s.c_str());
//   database->loadDatabase();
//   database->loadTestData();
//
//   return 0;
// }
