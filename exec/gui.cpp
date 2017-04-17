#include <imgui.h>
#include "imgui_impl_glfw_gl3.h"
#include <stdio.h>
#include <GL/gl3w.h>
#include <GLFW/glfw3.h>
#include <fstream>
#include <vector>
#include "gui.h"
#include "dirent.h"
#include <sys/stat.h>
#include <gflags/gflags.h>

#define IM_ARRAYSIZE(_ARR)  ((int)(sizeof(_ARR)/sizeof(*_ARR)))

#define GUI_TEST_WIDTH    (glfwDisplay.screen_h/2 - GUI_GAP_SIZE/2)
#define GUI_SETTING_WIDTH 400
#define GUI_GAP_SIZE      10

// GFLAGS
// DEFINE_bool: boolean
// DEFINE_int32: 32-bit integer
// DEFINE_int64: 64-bit integer
// DEFINE_uint64: unsigned 64-bit integer
// DEFINE_double: double
// DEFINE_string: C++ string

DEFINE_bool(batch_test, false, "do batch test");
DEFINE_string(dataset_root, "/Users/lgzhang/Documents/DATA/micro_gps_packed", "dataset_root");
DEFINE_string(dataset, "acee_asphalt_long_packed", "dataset to use");
DEFINE_string(testset, "test00.test", "test sequence");
DEFINE_string(output, "test_results", "output");
DEFINE_string(feature_database, "acee_asphalt_long_packed-siftgpu.bin", "database features");
DEFINE_string(pca_basis, "", "pca basis to use");
DEFINE_string(map, "acee_asphalt_map_10per.png", "stitched map");
DEFINE_double(map_scale, 0.1, "map scale");
DEFINE_double(cell_size, 50.0f, "size of the voting cell");
DEFINE_int32(num_scale_groups, 10, "number of search indexes");
DEFINE_int32(dimensionality, 8, "dimensionality after PCA reduction");
DEFINE_int32(best_knn, 9999, "use the best k nearest neighbors for voting");
DEFINE_double(sift_extraction_scale, 0.5, "extract sift at this scale");
DEFINE_bool(test_all, false, "test all frames");
DEFINE_bool(nogui, false, "disable gui");
// offline
DEFINE_int32(database_sample_size, 50, "number of features sampled from each database iamge");
// precomputed_feature
DEFINE_string(dataset_precomputed, "", "database images precomputed features - .txt");
DEFINE_string(test_precomputed, "", "test images precomputed features - .txt");

MicroGPSVariables mgpsVars;
GLFWDisplay glfwDisplay;
bool batch_test_initialized = false;
char robot_camera_calibration_file_path[] = "robot_camera_calibration.txt";
float mm_per_pixel;

void readRobotCameraCalibration (char* file_path) {
  Eigen::MatrixXf T_camera_robot = Eigen::MatrixXf::Identity(3, 3);
  FILE* fp = fopen(file_path, "r");
  fscanf(fp, "%f %f %f\n", &T_camera_robot(0, 0), &T_camera_robot(0, 1), &T_camera_robot(0, 2));
  fscanf(fp, "%f %f %f\n", &T_camera_robot(1, 0), &T_camera_robot(1, 1), &T_camera_robot(1, 2));
  fscanf(fp, "%f\n", &mm_per_pixel);
  fclose(fp);

  std::cout << "T_camera_robot = " << T_camera_robot << std::endl;
  std::cout << "mm_per_pixel = " << mm_per_pixel << std::endl;
}

int listDir(std::string dir, std::vector<std::string>& files, std::string keyword, bool fname_only = false) {
  files.clear();

  DIR* dp;
  struct dirent* dirp;
  if((dp  = opendir(dir.c_str())) == NULL) {
      return -1;
  }

  while ((dirp = readdir(dp)) != NULL) {
  	std::string name = std::string(dirp->d_name);
    // exceptions
  	if(name != "." && name != ".." && name != ".DS_Store") {
      if (keyword != ""){
        if (name.find(keyword) != std::string::npos) {
          files.push_back(name);
        }
      } else {
        files.push_back(name);
      }
    }
  }
  closedir(dp);

  // sort files
  std::sort(files.begin(), files.end());

  if(dir.at( dir.length() - 1 ) != '/') {
    dir = dir + "/";
  }

  for(unsigned int i = 0;i<files.size();i++) {
    if(files[i].at(0) != '/') {
      if (!fname_only) {
        files[i] = dir + files[i];
      }
    }
  }

  return files.size();
}

bool checkFileExists(const char* path) {
  struct stat buffer;   
  if (stat (path, &buffer) != 0) {
    return false;
  }
  return true;
}

void mkdirIfNotExists(char* path) {
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


WorkImageGL3Texture::WorkImageGL3Texture() {
  id = -1;
  width = -1.0f;
  height = -1.0f;
  show = false;
}

void WorkImageGL3Texture::loadTextureFromWorkImage(WorkImage* work_image, bool rotate90) {
  cv::Mat cvImg(work_image->height(), work_image->width(), CV_8UC3, work_image->data());
  this->loadTextureFromCVMat(cvImg, rotate90);
}

void WorkImageGL3Texture::loadTextureFromCVMat(cv::Mat& image, bool rotate90) {
  if (show) {
    this->disable();
  }

  cv::Mat image_to_load = image;
  rotated90 = rotate90;
  if (rotated90) {
    cv::transpose(image, image_to_load);
    cv::flip(image_to_load, image_to_load, 1);
  }

  glGenTextures(1, &id);

  glBindTexture(GL_TEXTURE_2D, id);

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // by default number of bytes in a row should be multiples of 4, causing problems
  // upload the image to OpenGL
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image_to_load.cols, image_to_load.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, image_to_load.data);

  printf("uploaded image\n");

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glGenerateMipmap(GL_TEXTURE_2D);


  width = image.cols; // still keep the origin size
  height = image.rows;

  show = true; // ready to display
}


void WorkImageGL3Texture::disable() {
  if (show) { // current being displayed
    glDeleteTextures(1, &id);
  }
  id = -1;
  width = -1.0f;
  height = -1.0f;
  show = false;
}

float WorkImageGL3Texture::getWidth() {
  if (rotated90) {
    return height;
  }
  return width;
}

float WorkImageGL3Texture::getHeight() {
  if (rotated90) {
    return width;
  }
  return height;
}


void MicroGPSVariables::loadDefaultValues() {
  //TODO: mkdir if necessary
#ifdef ON_MAC
  strcpy(dataset_root, "/Users/lgzhang/Documents/DATA/micro_gps_packed");
  strcpy(map_image_root, "maps");
  strcpy(database_root, "databases");
  strcpy(PCA_basis_root, "pca_bases");
  strcpy(screenshots_root, "screenshots");
  strcpy(test_results_root, "test_results");
#endif
#ifdef ON_AMD64
  strcpy(dataset_root, "/data/linguang/micro_gps_packed");
  strcpy(map_image_root, "maps");
  strcpy(database_root, "databases");
  strcpy(PCA_basis_root, "pca_bases");
  strcpy(screenshots_root, "screenshots");
  strcpy(test_results_root, "test_results");
#endif
#ifdef ON_TEGRA
  strcpy(dataset_root, "usbdrive/micro_gps_packed");
  strcpy(map_image_root, "usbdrive/maps");
  strcpy(database_root, "usbdrive/databases");
  strcpy(PCA_basis_root, "usbdrive/pca_bases");
  strcpy(screenshots_root, "usbdrive/screenshots");
  strcpy(test_results_root, "usbdrive/test_results");
#endif
  mkdirIfNotExists(test_results_root);

  // dataset_path.push_back("fc");
  dataset_path_selected = 0;
  // load_map_image_path.push_back("fc_map_10per.png");
  load_test_sequence_path_selected = -1;
  load_map_image_path_selected = -1;

  map_scales.resize(4);
  map_scales[0] = "10%";
  map_scales[1] = "25%";
  map_scales[2] = "50%";
  map_scales[3] = "100%";
  load_map_image_path_selected = 0;


  // Testing
  cell_size = 50.0f;
  load_database_path.push_back("cs4_hallway_siftgpu.bin");
  load_database_path_selected = -1;
  load_PCA_basis_path.push_back("cs4_hallway_pca_basis.bin");
  load_PCA_basis_path_selected = -1;
  test_index = 0;
  prev_test_index = 0;
  bool to_test_current_frame = false;
  bool to_save_tested_frame = false;


  // Training
  feature_sample_size = 50.0f;
  PCA_dimensions = 8;
  strcpy(save_map_image_path, "_map_10per.png");
  strcpy(save_database_path, "_siftgpu.bin");
  strcpy(save_PCA_basis_path, "_dims8_pca_basis.bin");


  // Monitor
  top_cells_histogram.resize(10);
  for (int i = 0; i < top_cells_histogram.size(); i++) {
    top_cells_histogram[i] = 0.0f;
  }
  success_flag = false;
  num_frames_tested = 0;
  num_frames_succeeded = 0;
  
  // Micro-GPS internals
  dataset = NULL;
  // enable_alignment = true;
  mgpsVars.options.do_alignment = true;
  mgpsVars.options.debug_mode = true;
  mgpsVars.options.confidence_thresh = 0.8f;

  show_test_window = false;
}

void generateMapFromDataset(Database* dataset, const char* output_path, float scale) {
  int n_images = mgpsVars.dataset->getDatabaseSize();
  // int n_images = 100;
  std::vector<WorkImage*> work_images(n_images);
  std::vector<Eigen::Matrix3f> work_image_poses(n_images);

  for (int i = 0; i < n_images; i++) {
    work_images[i] = new WorkImage(dataset->getDatabaseImage(i));
    work_image_poses[i] = dataset->getDatabasePose(i);
  }

  printf("all work images loaded\n");
  warpImageArray(work_images, work_image_poses, scale)->write(output_path);

  for (int i = 0; i < n_images; i++) {
    delete work_images[i];
  }
}

void computeMapOffsets(Database* dataset, float scale,
                      float& world_min_x_out, float& world_min_y_out) {
  int n_images = mgpsVars.dataset->getDatabaseSize();
  // int n_images = 100;
  std::vector<WorkImage*> work_images(n_images);
  std::vector<Eigen::Matrix3f> work_image_poses(n_images);

  for (int i = 0; i < n_images; i++) {
    work_images[i] = new WorkImage(dataset->getDatabaseImage(i));
    work_image_poses[i] = dataset->getDatabasePose(i);
  }

  int world_min_x;
  int world_min_y;
  int world_max_x;
  int world_max_y;

  computeImageArrayWorldSize(work_images,
                            work_image_poses,
                            scale,
                            world_min_x,
                            world_min_y,
                            world_max_x,
                            world_max_y);

  for (int i = 0; i < n_images; i++) {
    delete work_images[i];
  }

  world_min_x_out = world_min_x;
  world_min_y_out = world_min_y;
}

float globalLength2TextureLength(float x) {
  int percentage;
  sscanf(mgpsVars.map_scales[mgpsVars.load_map_scale_selected].c_str(), "%d%", &percentage);
  float ratio = ((float)percentage) / 100.0f;
  return x * ratio * mgpsVars.map_texture_info.fitting_scale;
}

void globalCoordinates2TextureCoordinates(float& x, float& y) {
  int percentage;
  sscanf(mgpsVars.map_scales[mgpsVars.load_map_scale_selected].c_str(), "%d%", &percentage);
  float ratio = ((float)percentage) / 100.0f;

  x = (x * ratio - mgpsVars.world_min_x) * mgpsVars.map_texture_info.fitting_scale;
  y = (y * ratio - mgpsVars.world_min_y) * mgpsVars.map_texture_info.fitting_scale;

}

// http://stackoverflow.com/questions/7706339/grayscale-to-red-green-blue-matlab-jet-color-scale
void gray2jet(double v,double vmin, double vmax,
                double& r, double& g, double& b) {
  r = 1.0f;
  g = 1.0f;
  b = 1.0f;
  double dv;

  if (v < vmin)
    v = vmin;
  if (v > vmax)
    v = vmax;
  dv = vmax - vmin;

  if (v < (vmin + 0.25 * dv)) {
    r = 0;
    g = 4 * (v - vmin) / dv;
  } else if (v < (vmin + 0.5 * dv)) {
    r = 0;
    b = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
  } else if (v < (vmin + 0.75 * dv)) {
    r = 4 * (v - vmin - 0.5 * dv) / dv;
    b = 0;
  } else {
    g = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
    b = 0;
  }
}

void generateMapDensity(std::vector<Eigen::Vector2f> points, WorkImageGL3Texture& texture, 
                        double& vmin_out, double& vmax_out,
                        double vmin_in = -1.0, double vmax_in = -1.0) {
  printf("generate map density\n");
  int w = round(mgpsVars.map_texture.width * mgpsVars.map_texture_info.fitting_scale);
  int h = round(mgpsVars.map_texture.height * mgpsVars.map_texture_info.fitting_scale);

  cv::Mat map = cv::Mat::zeros(h, w, CV_64FC1);
  int cnt = 0;
  for (size_t i = 0; i < points.size(); i++) {
    int x = (int)floor(points[i].x());
    int y = (int)floor(points[i].y());
    if (x < 0 || x > w-1 || y < 0 || y > h-1 ) {
      continue;
    }
    cnt ++;
    map.at<double>(y, x) += 1.0f;
  }

  cv::GaussianBlur(map, map, cv::Size(21, 21), 5.0f);

  double vmin, vmax;
  if (vmin_in >= 0.0 && vmax_in >= 0.0) {
    vmin = vmin_in;
    vmax = vmax_in;
    printf("vmax = %f, vmin = %f\n", vmax, vmin);
  } else {
    cv::minMaxLoc(map, &vmin, &vmax);
    vmin_out = vmin;
    vmax_out = vmax;
  }

  // printf("cnt = %d, vmin = %lf, vmax = %lf\n", cnt , vmin, vmax);

  cv::Mat map_jet(h, w, CV_64FC3);
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      double r,g,b;
      double val = map.at<double>(y, x);
      if (val < vmin) {
        val = vmin;
      }
      if (val > vmax) {
        val = vmax;
      }
      gray2jet(map.at<double>(y, x), vmin, vmax, r, g, b);
      map_jet.at<cv::Vec3d>(y, x)[0] = b;
      map_jet.at<cv::Vec3d>(y, x)[1] = g;
      map_jet.at<cv::Vec3d>(y, x)[2] = r;
    }
  }


  map_jet *= 255.0f;
  map_jet.convertTo(map_jet, CV_8UC3);

  // cv::imwrite("test_density.png", map_jet);
  texture.loadTextureFromCVMat(map_jet, mgpsVars.map_texture.rotated90);

}


void testCurrentFrame() {
  // convenient function for testing the current frame
  mgpsVars.success_flag = false;
  WorkImage* current_test_frame = new WorkImage(mgpsVars.dataset->getTestImage(mgpsVars.test_index),
                                                mgpsVars.dataset->getTestPrecomputedFeatures(mgpsVars.test_index));
  current_test_frame->loadImage();

  WorkImage* alignment_image = NULL;
  mgpsVars.timing.reset();
  mgpsVars.result.reset();
  mgpsVars.debug.reset();  

  mgpsVars.success_flag = mgpsVars.micro_gps->locate(current_test_frame, alignment_image,
                                                      mgpsVars.options, mgpsVars.result,
                                                      mgpsVars.timing, mgpsVars.debug);
  mgpsVars.result.success_flag = mgpsVars.success_flag;
  current_test_frame->release();
  delete current_test_frame;

  if (!FLAGS_nogui && mgpsVars.success_flag && mgpsVars.options.debug_mode) {
    if (mgpsVars.options.do_alignment) {
      mgpsVars.alignment_texture.loadTextureFromWorkImage(alignment_image);
      delete alignment_image;
    } else {
      mgpsVars.alignment_texture.disable();
    }
    // int percentage;
    // sscanf(mgpsVars.map_scales[mgpsVars.load_map_scale_selected].c_str(), "%d%", &percentage);
    // float ratio = ((float)percentage) / 100.0f;

    int n_candidates = std::min(3000, (int)mgpsVars.debug.candidate_image_poses.size());
    std::vector<Eigen::Vector2f> image_origins(n_candidates);
    for (int fidx = 0; fidx < n_candidates; fidx++) {
      float kp_x = mgpsVars.debug.candidate_image_poses[fidx](0, 2);
      float kp_y = mgpsVars.debug.candidate_image_poses[fidx](1, 2);
      globalCoordinates2TextureCoordinates(kp_x, kp_y);
      image_origins[fidx](0) = kp_x;
      image_origins[fidx](1) = kp_y;
    }


    double vmin_out, vmax_out;
    generateMapDensity(image_origins, mgpsVars.map_image_pose_overlay_texture,
                      vmin_out, vmax_out);

    printf("vmax_out = %f, vmin_out = %f\n", vmax_out, vmin_out);

    int n_keypoints = std::min(3000, (int)mgpsVars.debug.test_feature_poses.size());
    std::vector<Eigen::Vector2f> keypoints(n_keypoints);
    for (int fidx = 0; fidx < n_keypoints; fidx++) {
      float kp_x = mgpsVars.debug.knn_matched_feature_poses[fidx](0, 2);
      float kp_y = mgpsVars.debug.knn_matched_feature_poses[fidx](1, 2);
      globalCoordinates2TextureCoordinates(kp_x, kp_y);
      keypoints[fidx](0) = kp_x;
      keypoints[fidx](1) = kp_y;
    }

    double vmin_in = vmin_out;
    double vmax_in = vmax_out;
    generateMapDensity(keypoints, mgpsVars.map_feature_pose_overlay_texture,
                      vmin_out, vmax_out,
                      vmin_in, vmax_in);

  } else {
    mgpsVars.alignment_texture.disable();
    mgpsVars.map_feature_pose_overlay_texture.disable();
    mgpsVars.map_image_pose_overlay_texture.disable();
  }

  printf("Finish testing, generating log\n");

  char test_report_path[256];
  sprintf(test_report_path, "%s/frame%06d.txt", mgpsVars.test_results_root, mgpsVars.test_index);
  FILE* fp = fopen(test_report_path, "w");
  mgpsVars.timing.printToFile(fp);
  mgpsVars.result.printToFile(fp);
  mgpsVars.debug.printToFile(fp);
  fclose(fp);
}


// glReadPixels(	GLint x,
//  	GLint y,
//  	GLsizei width,
//  	GLsizei height,
//  	GLenum format,
//  	GLenum type,
//  	GLvoid * data);

void saveGUIRegion(int topleft_x, int topleft_y, int width, int height,
                  const char* out_path) {

  int multiplier_x = glfwDisplay.framebuffer_w / glfwDisplay.screen_w;
  int multiplier_y = glfwDisplay.framebuffer_h / glfwDisplay.screen_h;

  // we just read RGB
  uchar* data = new uchar[height * multiplier_y * width * multiplier_x * 3];

  int lowerleft_x = topleft_x;
  int lowerleft_y = glfwDisplay.screen_h - (topleft_y + height);

  glReadBuffer(GL_FRONT); // wth is GL_FRONT_LEFT / GL_FRONT???
  glPixelStorei(GL_PACK_ALIGNMENT, 1); // fixing the "multiples of 4" problem
  glReadPixels(lowerleft_x * multiplier_x, lowerleft_y * multiplier_y,
   	          width * multiplier_x, height * multiplier_y,
             	GL_BGR,
             	GL_UNSIGNED_BYTE,
             	data);

  // wrap it
  cv::Mat bgr(height*multiplier_y, width*multiplier_x, CV_8UC3, data);
  cv::flip(bgr, bgr, 0);

  char s[256];
  sprintf(s, "%s/%s", mgpsVars.test_results_root, out_path);
  cv::imwrite(s, bgr);

  delete[] data;
}


namespace ImGui {
  bool Combo(const char* label, int* current_item, const std::vector<std::string>& items, int height_in_items = -1);
}

bool ImGui::Combo(const char* label, int* current_item, const std::vector<std::string>& items, int height_in_items) {
  return Combo(label, current_item,
        [](void* data, int idx, const char** out_text) {
            *out_text = (*(const std::vector<std::string>*)data)[idx].c_str();
            return true;
          },
        (void*)&items, items.size(), height_in_items);
}


static void error_callback(int error, const char* description) {
  fprintf(stderr, "Error %d: %s\n", error, description);
}


float computeWindowHeight(int num_items, int num_texts, float others = 0.0f) {
  float spacing = (float)(num_items + num_texts - 1) * ImGui::GetStyle().ItemSpacing.y;
  float items_height = (float)num_items * (ImGui::GetWindowFontSize() + ImGui::GetStyle().FramePadding.y*2);
  float texts_height = (float)num_texts * ImGui::GetWindowFontSize();
  return  items_height +
          texts_height +
          spacing +
          others+
          ImGui::GetStyle().ItemSpacing.y;
          // ImGui::GetStyle().WindowPadding.y*2;
}

void eventLoadMap(const char* map_name, double map_scale) {
  char s[256];
  sprintf(s, "%s/%s", mgpsVars.map_image_root, map_name);
  if (!checkFileExists(s)) {
    generateMapFromDataset(mgpsVars.dataset, s, map_scale);
  }

  WorkImage* new_map = new WorkImage(s);
  new_map->loadImage();
  bool rotate90 = false;
  if (mgpsVars.map_texture_avail_height > mgpsVars.map_texture_avail_width != 
      new_map->height() > new_map->width()) {
    rotate90 = true;
  }
  mgpsVars.map_texture.loadTextureFromWorkImage(new_map, rotate90);
  delete new_map;

  computeMapOffsets(mgpsVars.dataset, map_scale,
                    mgpsVars.world_min_x, mgpsVars.world_min_y);

}

void eventLoadDataset(const char* dataset_path) {
  if (mgpsVars.dataset != NULL) {
    delete mgpsVars.dataset;
  }

  mgpsVars.dataset = new Database(dataset_path);
  mgpsVars.dataset->loadDatabase();
  if (FLAGS_dataset_precomputed != "") {
    printf("loading database images precomputed features\n");
    mgpsVars.dataset->loadDatabasePrecomputedFeatures(FLAGS_dataset_precomputed.c_str());
  }
  if (FLAGS_test_precomputed != "") {
    printf("loading test images precomputed features\n");
    mgpsVars.dataset->loadTestSequencePrecomputedFeatures(FLAGS_test_precomputed.c_str());    
  }


  mgpsVars.micro_gps = new MicroGPS();
  mgpsVars.micro_gps->loadDatabaseOnly(mgpsVars.dataset);

  printf("loaded dataset\n");

  // reset visualzation
  mgpsVars.map_texture.disable();
  mgpsVars.test_image_texture.disable();
  mgpsVars.alignment_texture.disable();
  mgpsVars.map_feature_pose_overlay_texture.disable();
  mgpsVars.map_image_pose_overlay_texture.disable();
  mgpsVars.success_flag = false;
}

void eventInitMicroGPS(double cell_size, int num_scale_groups, const char* database_name, const char* pca_basis_name, int dimensionality) {
  mgpsVars.micro_gps->setVotingCellSize(cell_size);
  mgpsVars.micro_gps->setNumScaleGroups(num_scale_groups);
  mgpsVars.micro_gps->loadDatabaseOnly(mgpsVars.dataset);

  // reload precomputed values
  char s[256];
  sprintf(s, "%s/%s", mgpsVars.database_root, database_name);
  if (!checkFileExists(s)) { // create if not exists
    mgpsVars.micro_gps->preprocessDatabaseImages(FLAGS_database_sample_size, FLAGS_sift_extraction_scale);
    mgpsVars.micro_gps->saveFeatures(s);
  }
  
  mgpsVars.micro_gps->loadFeatures(s);
  if (strcmp(pca_basis_name, "") != 0) {
    sprintf(s, "%s/%s", mgpsVars.PCA_basis_root, pca_basis_name);
    mgpsVars.micro_gps->loadPCABasis(s);
  } else {
    mgpsVars.micro_gps->computePCABasis();
    sprintf(s, "%s/pca_%s", mgpsVars.PCA_basis_root, database_name);
    mgpsVars.micro_gps->savePCABasis(s);
  }
  mgpsVars.micro_gps->PCAreduction(dimensionality);
  mgpsVars.micro_gps->buildSearchIndexMultiScales();
}

void eventTestAll (bool trigger = false) {
  static bool to_save = false;
  static bool to_test = false;

  if (trigger) {
    to_save = false;
    to_test = true;    
    mgpsVars.num_frames_tested = 0;
    mgpsVars.num_frames_succeeded = 0;
    return;
  }

  static char s[256];
  // save screenshots after the previous frame is rendered
  if (to_save) {
    if (!FLAGS_nogui) {
      saveGUIRegion((int)mgpsVars.map_texture_info.screen_pos_x+1, (int)mgpsVars.map_texture_info.screen_pos_y,
                    (int)mgpsVars.map_texture_info.width, (int)mgpsVars.map_texture_info.height,
                    s);
    }
    if (mgpsVars.test_index < mgpsVars.dataset->getTestSize() - 1) {
      mgpsVars.test_index++;
      to_test = true;
    } else if (FLAGS_batch_test) {
      exit(0);
    }
    to_save = false;
  }

  if (to_test) {
    testCurrentFrame();
    sprintf(s, "frame%06d_map.png", mgpsVars.test_index);
    to_save = true;
    to_test = false;
    mgpsVars.num_frames_tested++;
    mgpsVars.num_frames_succeeded += mgpsVars.success_flag;
  } 
}

void eventPrintResults() {
  ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);


  ImGui::Text("Test result: %s (%d/%d) = %f%%", mgpsVars.success_flag ? "SUCCESS" : "FAILURE", mgpsVars.num_frames_succeeded, mgpsVars.num_frames_tested,
                                              (float)mgpsVars.num_frames_succeeded / (float)mgpsVars.num_frames_tested * 100.0f);
  ImGui::Text("dx=%.02f, dy=%.02f da=%.02f\n", mgpsVars.result.x_error, mgpsVars.result.y_error, mgpsVars.result.angle_error);
  ImGui::Text("Total time: %.3lf ms", mgpsVars.timing.total * 1000.0f);
  ImGui::BulletText("SIFT extraction : %.3lf ms", mgpsVars.timing.sift_extraction * 1000.0f);
  ImGui::BulletText("KNN search : %.3lf ms", mgpsVars.timing.knn_search * 1000.0f);
  ImGui::BulletText("Compute candidate poses : %.3lf ms", mgpsVars.timing.candidate_image_pose * 1000.0f);
  ImGui::BulletText("Voting : %.3lf ms", mgpsVars.timing.voting * 1000.0f);
  ImGui::BulletText("RANSAC : %.3lf ms", mgpsVars.timing.ransac * 1000.0f);

  if (mgpsVars.result.top_cells.size() > 0) {
    int hist_max = *max_element(mgpsVars.result.top_cells.begin(), mgpsVars.result.top_cells.end());
    int n_cells = mgpsVars.result.top_cells.size();
    float* data = new float[n_cells];
    for (int i = 0; i < n_cells; i++) {
      data[i] = (float)mgpsVars.result.top_cells[i];
    }

    ImGui::Columns(n_cells, NULL, true);
    ImGui::Separator();
    for (int i = 0; i < n_cells; i++) {
        ImGui::Text("%.0f", data[i]);
        ImGui::NextColumn();
    }
    ImGui::Columns(1);
    ImGui::Separator();
  }

}


void drawBatchTestMonitor() {
  if (!batch_test_initialized) {
    return;
  }
  ImGui::SetNextWindowSize(ImVec2(GUI_SETTING_WIDTH,glfwDisplay.screen_h));
  ImGui::SetNextWindowPos(ImVec2(glfwDisplay.screen_w-GUI_SETTING_WIDTH, 0));

  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
  ImGui::Begin("Batch", NULL,  ImGuiWindowFlags_NoCollapse|
                                  ImGuiWindowFlags_NoResize|
                                  ImGuiWindowFlags_NoMove);


  // ImGui::SetNextTreeNodeOpen(true, ImGuiSetCond_Once);

  // ImGui::Text("dataset root =\n%s\n\n", FLAGS_dataset_root.c_str());
  ImGui::Text("dataset =\n%s\n\n", FLAGS_dataset.c_str());
  ImGui::Text("test set =\n%s\n\n", FLAGS_testset.c_str());
  ImGui::Text("dataset =\n%s\n\n", FLAGS_dataset.c_str());
  ImGui::Text("feature database =\n%s\n\n", FLAGS_feature_database.c_str());
  ImGui::Text("map =\n%s\n\n", FLAGS_map.c_str());
  ImGui::Text("map_scale      = %f\n\n", FLAGS_map_scale);
  ImGui::Text("test_all       = %d\n\n", FLAGS_test_all);
  ImGui::Text("output =\n%s\n\n", FLAGS_output.c_str());
  ImGui::Text("cell size      = %f\n", FLAGS_cell_size);
  ImGui::Text("# scale groups = %d\n", FLAGS_num_scale_groups);
  ImGui::Text("dimensionality = %d\n", FLAGS_dimensionality);
  ImGui::Text("best knn = %d\n", FLAGS_best_knn);
  ImGui::Text("sift extraction scale = %f\n\n", FLAGS_sift_extraction_scale);
  
  if (mgpsVars.test_index != mgpsVars.prev_test_index) {
    WorkImage* current_test_frame = new WorkImage(mgpsVars.dataset->getTestImage(mgpsVars.test_index));
    current_test_frame->loadImage();
    mgpsVars.test_image_texture.loadTextureFromWorkImage(current_test_frame);
    delete current_test_frame;
    mgpsVars.prev_test_index = mgpsVars.test_index;
  }




  if (ImGui::Button("test all") || FLAGS_test_all) {
    eventTestAll(true);
    FLAGS_test_all = false;
  } else {
    eventTestAll(false);
  }

  
  ImGui::Text("testing %d / %d\n", mgpsVars.test_index+1, mgpsVars.dataset->getTestSize());
  eventPrintResults();
  
  ImGui::End();
}




void drawSetting() {
  ImGui::SetNextWindowSize(ImVec2(GUI_SETTING_WIDTH,glfwDisplay.screen_h));
  ImGui::SetNextWindowPos(ImVec2(glfwDisplay.screen_w-GUI_SETTING_WIDTH, 0));

  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
  ImGui::Begin("Settings", NULL,  ImGuiWindowFlags_NoCollapse|
                                  ImGuiWindowFlags_NoResize|
                                  ImGuiWindowFlags_NoMove);

  ImGui::SetNextTreeNodeOpen(true, ImGuiSetCond_Once);
  if (ImGui::CollapsingHeader("Dataset")){

    listDir(mgpsVars.dataset_root, mgpsVars.dataset_path, "", true); // list datasets
    
    ImGui::Combo("###dataset_name", &mgpsVars.dataset_path_selected, mgpsVars.dataset_path);
    ImGui::SameLine();

    char selected_dataset_path[256];
    sprintf(selected_dataset_path, "%s/%s", mgpsVars.dataset_root, mgpsVars.dataset_path[mgpsVars.dataset_path_selected].c_str());
    if (ImGui::Button("load dataset", ImVec2(-1, 0))) {
      eventLoadDataset(selected_dataset_path);
    }

    listDir(selected_dataset_path, mgpsVars.load_test_sequence_path, ".test", true); // list test sequences
    ImGui::Combo("###load_test_sequence_path", &mgpsVars.load_test_sequence_path_selected, mgpsVars.load_test_sequence_path);
    ImGui::SameLine();
    if (ImGui::Button("load test", ImVec2(-1, 0))) {
      mgpsVars.dataset->loadTestSequenceByName(mgpsVars.load_test_sequence_path[mgpsVars.load_test_sequence_path_selected].c_str());
    }


    listDir(mgpsVars.map_image_root, mgpsVars.load_map_image_path, "", true); // list maps
    ImGui::Combo("map path###load_map_path", &mgpsVars.load_map_image_path_selected, mgpsVars.load_map_image_path);

    // selection of scale
    // static int load_map_scale_selected = 0;
    ImGui::Combo("map scale###load_map_scale", &mgpsVars.load_map_scale_selected, mgpsVars.map_scales);


    // ImGui::SameLine();
    if (ImGui::Button("load map", ImVec2(-1, 0))) {
      int percentage;
      sscanf(mgpsVars.map_scales[mgpsVars.load_map_scale_selected].c_str(), "%d%", &percentage);
      eventLoadMap(mgpsVars.load_map_image_path[mgpsVars.load_map_image_path_selected].c_str(), (double)percentage / 100.0f);
    }
  }

  ImGui::SetNextTreeNodeOpen(true, ImGuiSetCond_Once);
  if (ImGui::CollapsingHeader("Testing") ) {
    
    ImGui::InputFloat("cell size", &mgpsVars.cell_size, 1.0f, 0.0f, 1);
    if (mgpsVars.cell_size <= 0) { // set some limit
      mgpsVars.cell_size = 1.0f;
    }
    ImGui::InputInt("scale groups", &mgpsVars.scale_groups);
    if (mgpsVars.scale_groups <= 0) { // set some limit
      mgpsVars.scale_groups = 1;
    }

    ImGui::InputInt("dimensionality", &mgpsVars.PCA_dimensions);
    if (mgpsVars.PCA_dimensions <= 0) { // set some limit
      mgpsVars.PCA_dimensions = 1;
    }

    listDir(mgpsVars.database_root, mgpsVars.load_database_path, "", true); // list databases
    ImGui::Combo("database", &mgpsVars.load_database_path_selected, mgpsVars.load_database_path);
    listDir(mgpsVars.PCA_basis_root, mgpsVars.load_PCA_basis_path, "", true); // list PCA bases
    ImGui::Combo("PCA basis", &mgpsVars.load_PCA_basis_path_selected, mgpsVars.load_PCA_basis_path);

    const char* selected_pca_basis_path = NULL;
    if (mgpsVars.load_PCA_basis_path_selected >= 0 && 
        mgpsVars.load_PCA_basis_path_selected < mgpsVars.load_PCA_basis_path.size()) {
      selected_pca_basis_path = mgpsVars.load_PCA_basis_path[mgpsVars.load_PCA_basis_path_selected].c_str();
    }

    if (ImGui::Button("reload", ImVec2(-1, 0))) {
      eventInitMicroGPS(mgpsVars.cell_size, mgpsVars.scale_groups, 
                        mgpsVars.load_database_path[mgpsVars.load_database_path_selected].c_str(),
                        selected_pca_basis_path,
                        mgpsVars.PCA_dimensions);
    }

    int max_test_index = 999;
    if (mgpsVars.dataset) {
      max_test_index = mgpsVars.dataset->getTestSize()-1;
    }
    ImGui::SliderInt("image index", &mgpsVars.test_index, 0, max_test_index);
    // ImGui::PushItemWidth(30);
    // ImGui::SameLine();
    if (ImGui::Button("-", ImVec2(30, 0))) {
      mgpsVars.test_index--;
    }
    ImGui::SameLine();
    if (ImGui::Button("+", ImVec2(30, 0))) {
      mgpsVars.test_index++;
    }
    ImGui::SameLine();
    static char test_all_prefix[256] = "prefix";
    ImGui::PushItemWidth(100);
    ImGui::InputText("###test_all_prefix", test_all_prefix, 256);
    ImGui::PopItemWidth();
    ImGui::SameLine();
    static bool to_save = false;
    static bool to_test = false;
    if (ImGui::Button("test all")) {
      eventTestAll(true);
    }
    ImGui::Checkbox("Debug", &mgpsVars.options.debug_mode);
    ImGui::SameLine();
    ImGui::Checkbox("Alignment", &mgpsVars.options.do_alignment);
    ImGui::SameLine();
    ImGui::Checkbox("SIFT-Match", &mgpsVars.options.do_siftmatch);

    eventTestAll(false);

    if (mgpsVars.test_index != mgpsVars.prev_test_index) {
      WorkImage* current_test_frame = new WorkImage(mgpsVars.dataset->getTestImage(mgpsVars.test_index));
      current_test_frame->loadImage();
      mgpsVars.test_image_texture.loadTextureFromWorkImage(current_test_frame);
      delete current_test_frame;
      mgpsVars.prev_test_index = mgpsVars.test_index;
    }

    ImGui::InputFloat("sift scale", &mgpsVars.options.image_scale_for_sift, 0.05f, 0.0f, 2);
    if (mgpsVars.options.image_scale_for_sift <= 0.0) { // set some limit
      mgpsVars.options.image_scale_for_sift = 0.05;
    } else if (mgpsVars.options.image_scale_for_sift > 1.0) {
      mgpsVars.options.image_scale_for_sift = 1.0;      
    }

    ImGui::InputInt("best kNN", &mgpsVars.options.best_knn);
    if (mgpsVars.options.best_knn <= 0) { // set some limit
      mgpsVars.options.best_knn = 1;
    }

    if (ImGui::Button("locate", ImVec2(-1, 0))) {
      testCurrentFrame();
    }
  }

  ImGui::SetNextTreeNodeOpen(true, ImGuiSetCond_Once);
  if (ImGui::CollapsingHeader("Training")) {

    ImGui::InputText("map path", mgpsVars.save_map_image_path, 256);
    static int save_map_scale_selected = 0;

    ImGui::Combo("map scale###save", &save_map_scale_selected, mgpsVars.map_scales);

    if (ImGui::Button("generate map", ImVec2(-1, 0))) {
      char s[256];
      sprintf(s, "%s/%s", mgpsVars.map_image_root, mgpsVars.save_map_image_path);
      int percentage;
      sscanf(mgpsVars.map_scales[save_map_scale_selected].c_str(), "%d%", &percentage);
      generateMapFromDataset(mgpsVars.dataset, s, (float)percentage / 100.0f);
    }
   
    ImGui::InputInt("sample size", &mgpsVars.feature_sample_size);
    if (mgpsVars.feature_sample_size <= 0) { // set some limit
      mgpsVars.feature_sample_size = 1;
    }

    ImGui::InputText("database###save_database", mgpsVars.save_database_path, 256);
    ImGui::InputText("PCA basis###save_pca_basis", mgpsVars.save_PCA_basis_path, 256);

    if (ImGui::Button("process", ImVec2(-1, 0))) {
      mgpsVars.micro_gps->preprocessDatabaseImages(mgpsVars.feature_sample_size, mgpsVars.options.image_scale_for_sift);
      mgpsVars.micro_gps->computePCABasis();
      mgpsVars.micro_gps->PCAreduction(mgpsVars.PCA_dimensions);
      char s[256];
      sprintf(s, "%s/%s", mgpsVars.PCA_basis_root, mgpsVars.save_PCA_basis_path);
      mgpsVars.micro_gps->savePCABasis(s);
      sprintf(s, "%s/%s", mgpsVars.database_root, mgpsVars.save_database_path);
      mgpsVars.micro_gps->saveFeatures(s);
      // mgpsVars.micro_gps->buildSearchIndex();
      // mgpsVars.micro_gps->buildSearchIndexMultiScales();
    }
  }


  ImGui::SetNextTreeNodeOpen(true, ImGuiSetCond_Once);
  if (ImGui::CollapsingHeader("Monitor")) {

    ImVec2 size = ImGui::GetContentRegionAvail();
   
    eventPrintResults();

    if (ImGui::Button("demo")) {
      mgpsVars.show_test_window ^= 1;
    }

    if (mgpsVars.show_test_window) {
      ImGui::SetNextWindowPos(ImVec2(650, 20), ImGuiSetCond_FirstUseEver);
      ImGui::ShowTestWindow(&mgpsVars.show_test_window);
    }
  }

  ImGui::End();
  ImGui::PopStyleVar();

}

void drawTestImage() {
  ImGui::SetNextWindowSize(ImVec2(GUI_TEST_WIDTH, GUI_TEST_WIDTH));
  ImGui::SetNextWindowPos(ImVec2(glfwDisplay.screen_w-GUI_TEST_WIDTH-GUI_SETTING_WIDTH-GUI_GAP_SIZE, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
  ImGui::Begin("Test Image", NULL,  ImGuiWindowFlags_NoCollapse|
                                    ImGuiWindowFlags_NoResize|
                                    ImGuiWindowFlags_NoMove|
                                    ImGuiWindowFlags_NoScrollbar);


  ImVec2 region_avail = ImGui::GetContentRegionAvail(); // exluding padding
  region_avail.y -= ImGui::GetItemsLineHeightWithSpacing();  // reserving space for other widgets

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(2.0f, 2.0f));
  ImGui::BeginChild("test_image_disp", ImVec2(region_avail.x, region_avail.y), true);

  ImVec2 tex_screen_pos;
  float tex_w;
  float tex_h;

  // display the image
  if (mgpsVars.test_image_texture.show) {
    // compute texture size
    ImVec2 im_disp_region = ImGui::GetContentRegionAvail();
    float im_scaling = std::min(im_disp_region.x / mgpsVars.test_image_texture.width, im_disp_region.y / mgpsVars.test_image_texture.height);
    tex_w = mgpsVars.test_image_texture.width * im_scaling;
    tex_h = mgpsVars.test_image_texture.height * im_scaling;

    if (im_disp_region.x > tex_w) {
      ImGui::Indent((im_disp_region.x - tex_w) / 2.0f);
    }

    tex_screen_pos = ImGui::GetCursorScreenPos();
    // ImGui::Image((void*)mgpsVars.test_image_texture.id, ImVec2(tex_w, tex_h), ImVec2(0,0), ImVec2(1,1), ImColor(255,255,255,255), ImColor(0,0,0,0));


    ImGui::GetWindowDrawList()->AddImage((void*)mgpsVars.test_image_texture.id, ImVec2(tex_screen_pos.x, tex_screen_pos.y),
                                        ImVec2(tex_screen_pos.x+tex_w, tex_screen_pos.y+tex_h));

  }

  ImGui::EndChild();
  ImGui::PopStyleVar();


  static char save_rendered_test_image_path[256];
  ImGui::PushItemWidth(250);
  ImGui::InputText("###save_rendered_test_image_path", save_rendered_test_image_path, 256);
  ImGui::PopItemWidth();
  ImGui::SameLine();

  if (ImGui::Button("save")) {
    // printf("save image\n");
    saveGUIRegion((int)tex_screen_pos.x+1, (int)tex_screen_pos.y, (int)tex_w, (int)tex_h,
                  save_rendered_test_image_path);
  }


  ImGui::End();
  ImGui::PopStyleVar();
}


void drawAlignment() {
  ImGui::SetNextWindowSize(ImVec2(GUI_TEST_WIDTH, GUI_TEST_WIDTH));
  ImGui::SetNextWindowPos(ImVec2(glfwDisplay.screen_w-GUI_TEST_WIDTH-GUI_SETTING_WIDTH-GUI_GAP_SIZE, GUI_TEST_WIDTH+GUI_GAP_SIZE));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
  ImGui::Begin("Alignment", NULL,   ImGuiWindowFlags_NoCollapse|
                                    ImGuiWindowFlags_NoResize|
                                    ImGuiWindowFlags_NoMove|
                                    ImGuiWindowFlags_NoScrollbar);

  ImVec2 region_avail = ImGui::GetContentRegionAvail(); // exluding padding
  region_avail.y -= ImGui::GetItemsLineHeightWithSpacing();  // reserving space for other widgets

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(2.0f, 2.0f));
  ImGui::BeginChild("test_image_disp", ImVec2(region_avail.x, region_avail.y), true);


  ImVec2 tex_screen_pos;
  float tex_w;
  float tex_h;

  // display the image
  if (mgpsVars.alignment_texture.show) {
    // compute texture size
    ImVec2 im_disp_region = ImGui::GetContentRegionAvail();
    float im_scaling = std::min(im_disp_region.x / mgpsVars.alignment_texture.width, im_disp_region.y / mgpsVars.alignment_texture.height);
    tex_w = mgpsVars.alignment_texture.width * im_scaling;
    tex_h = mgpsVars.alignment_texture.height * im_scaling;

    if (im_disp_region.x > tex_w) {
      ImGui::Indent((im_disp_region.x - tex_w) / 2.0f);
    }

    tex_screen_pos = ImGui::GetCursorScreenPos();
    ImGui::Image((void*)mgpsVars.alignment_texture.id, ImVec2(tex_w, tex_h), ImVec2(0,0), ImVec2(1,1), ImColor(255,255,255,255), ImColor(0,0,0,0));

    // if (ImGui::IsItemHovered()) {
    //   ImGui::BeginTooltip();
    //   float focus_sz = 32.0f;
    //   float focus_x = ImGui::GetMousePos().x - tex_screen_pos.x - focus_sz * 0.5f; if (focus_x < 0.0f) focus_x = 0.0f; else if (focus_x > tex_w - focus_sz) focus_x = tex_w - focus_sz;
    //   float focus_y = ImGui::GetMousePos().y - tex_screen_pos.y - focus_sz * 0.5f; if (focus_y < 0.0f) focus_y = 0.0f; else if (focus_y > tex_h - focus_sz) focus_y = tex_h - focus_sz;
    //   ImGui::Text("Min: (%.2f, %.2f)", focus_x, focus_y);
    //   ImGui::Text("Max: (%.2f, %.2f)", focus_x + focus_sz, focus_y + focus_sz);
    //   ImVec2 uv0 = ImVec2((focus_x) / tex_w, (focus_y) / tex_h);
    //   ImVec2 uv1 = ImVec2((focus_x + focus_sz) / tex_w, (focus_y + focus_sz) / tex_h);
    //   ImGui::Image((void*)mgpsVars.alignment_texture.id, ImVec2(256,256), uv0, uv1, ImColor(255,255,255,255), ImColor(255,255,255,128));
    //   ImGui::EndTooltip();
    // }
  }

  ImGui::EndChild();
  ImGui::PopStyleVar();

  static char save_rendered_alignment_path[256];
  ImGui::PushItemWidth(250);
  ImGui::InputText("###save_rendered_alignment_path", save_rendered_alignment_path, 256);
  ImGui::PopItemWidth();
  ImGui::SameLine();

  if (ImGui::Button("save")) {
    saveGUIRegion((int)tex_screen_pos.x+1, (int)tex_screen_pos.y, (int)tex_w, (int)tex_h,
                  save_rendered_alignment_path);
  }


  ImGui::SameLine();

  ImGui::End();
  ImGui::PopStyleVar();
}


void drawMap() {
  ImGui::SetNextWindowSize(ImVec2(glfwDisplay.screen_w-GUI_TEST_WIDTH-GUI_SETTING_WIDTH-GUI_GAP_SIZE*2,glfwDisplay.screen_h));
  ImGui::SetNextWindowPos(ImVec2(0, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
  ImGui::Begin("Map", NULL,  ImGuiWindowFlags_NoCollapse|
                                    ImGuiWindowFlags_NoResize|
                                    ImGuiWindowFlags_NoMove|
                                    ImGuiWindowFlags_NoScrollbar);

  ImVec2 region_avail = ImGui::GetContentRegionAvail(); // exluding padding
  region_avail.y -= ImGui::GetItemsLineHeightWithSpacing() * 2;  // reserving space for other widgets

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(2.0f, 2.0f));
  ImGui::BeginChild("map_drawing_region", ImVec2(region_avail.x, region_avail.y), true);

  ImVec2 im_disp_region = ImGui::GetContentRegionAvail();
  mgpsVars.map_texture_avail_width = im_disp_region.x;
  mgpsVars.map_texture_avail_height = im_disp_region.y;;


  static int overlay_transparency = 128;
  static int overlay_idx = 1;

  // display the image
  if (mgpsVars.map_texture.show) {
    // compute texture size
    mgpsVars.map_texture_info.fitting_scale = std::min(mgpsVars.map_texture_avail_width / mgpsVars.map_texture.getWidth(), mgpsVars.map_texture_avail_height / mgpsVars.map_texture.getHeight());
    float tex_w = mgpsVars.map_texture.getWidth() * mgpsVars.map_texture_info.fitting_scale;
    float tex_h = mgpsVars.map_texture.getHeight() * mgpsVars.map_texture_info.fitting_scale;
    
    if (mgpsVars.map_texture_avail_width > tex_w) {
      ImGui::Indent((mgpsVars.map_texture_avail_width - tex_w) / 2.0f);
    }
    ImVec2 tex_screen_pos = ImGui::GetCursorScreenPos();

    // update texture info
    mgpsVars.map_texture_info.width = tex_w;
    mgpsVars.map_texture_info.height = tex_h;
    mgpsVars.map_texture_info.screen_pos_x = tex_screen_pos.x;
    mgpsVars.map_texture_info.screen_pos_y = tex_screen_pos.y;


    ImGui::Image((void*)mgpsVars.map_texture.id, ImVec2(tex_w, tex_h), ImVec2(0,0), ImVec2(1,1), ImColor(255,255,255,255), ImColor(0,0,0,0));
    ImGui::SetCursorScreenPos(tex_screen_pos); // go back


    switch (overlay_idx) {
      case 0:
        if (mgpsVars.map_feature_pose_overlay_texture.show) {
          ImGui::Image((void*)mgpsVars.map_feature_pose_overlay_texture.id, ImVec2(tex_w, tex_h), ImVec2(0,0), ImVec2(1,1), ImColor(255,255,255,overlay_transparency), ImColor(0,0,0,0));
        }
        break;
      case 1:
        if (mgpsVars.map_image_pose_overlay_texture.show) {
          ImGui::Image((void*)mgpsVars.map_image_pose_overlay_texture.id, ImVec2(tex_w, tex_h), ImVec2(0,0), ImVec2(1,1), ImColor(255,255,255,overlay_transparency), ImColor(0,0,0,0));
        }
        break;
    }

    // draw estimated location / orientation
    if (mgpsVars.success_flag) {
      float center_x, center_y;
      float x_axis_x, x_axis_y;
      float y_axis_x, y_axis_y;
      
      center_x = mgpsVars.result.final_estimated_pose(0, 2);
      center_y = mgpsVars.result.final_estimated_pose(1, 2);
      globalCoordinates2TextureCoordinates(center_x, center_y);

      if (mgpsVars.map_texture.rotated90) {
        float tmp = center_x;
        center_x = mgpsVars.map_texture_info.width - center_y;
        center_y = tmp;
        x_axis_y = mgpsVars.result.final_estimated_pose(0, 0);
        x_axis_x = -mgpsVars.result.final_estimated_pose(1, 0);
        y_axis_y = mgpsVars.result.final_estimated_pose(0, 1);
        y_axis_x = -mgpsVars.result.final_estimated_pose(1, 1);        
      } else {
        x_axis_x = mgpsVars.result.final_estimated_pose(0, 0);
        x_axis_y = mgpsVars.result.final_estimated_pose(1, 0);
        y_axis_x = mgpsVars.result.final_estimated_pose(0, 1);
        y_axis_y = mgpsVars.result.final_estimated_pose(1, 1);
      }

      center_x += tex_screen_pos.x;
      center_y += tex_screen_pos.y;

      if (mgpsVars.draw_camera) {
        ImGui::GetWindowDrawList()->AddLine(ImVec2(center_x, center_y), ImVec2(center_x + x_axis_x*15, center_y + x_axis_y*15),
                                              ImColor(0,0,255,255), 2.0f);
        ImGui::GetWindowDrawList()->AddLine(ImVec2(center_x, center_y), ImVec2(center_x + y_axis_x*15, center_y + y_axis_y*15),
                                              ImColor(255,0,0,255), 2.0f);
        ImGui::GetWindowDrawList()->AddCircle(ImVec2(center_x, center_y), 15, ImColor(0,255,0,255), 12, 2.0f);

        // float frame_width = globalLength2TextureLength(1288);
        // float frame_height = globalLength2TextureLength(964);
        // ImGui::GetWindowDrawList()->AddLine(ImVec2(center_x, center_y), ImVec2(center_x + x_axis_x*frame_width, center_y + x_axis_y*frame_width),
        //                                       ImColor(255,0,0,255), 2.0f);
        // ImGui::GetWindowDrawList()->AddLine(ImVec2(center_x, center_y), ImVec2(center_x + y_axis_x*frame_height, center_y + y_axis_y*frame_height),
        //                                       ImColor(0,0,255,255), 2.0f);
        // ImGui::GetWindowDrawList()->AddLine(ImVec2(center_x + x_axis_x*frame_width + y_axis_x*frame_height, center_y + y_axis_y*frame_height + x_axis_y*frame_width), ImVec2(center_x + x_axis_x*frame_width, center_y + x_axis_y*frame_width),
        //                                       ImColor(255,0,0,255), 2.0f);
        // ImGui::GetWindowDrawList()->AddLine(ImVec2(center_x + x_axis_x*frame_width + y_axis_x*frame_height, center_y + y_axis_y*frame_height + x_axis_y*frame_width), ImVec2(center_x + y_axis_x*frame_height, center_y + y_axis_y*frame_height),
        //                                       ImColor(0,0,255,255), 2.0f);

      }
    }


    // if (ImGui::IsItemHovered()) {
    //   ImGui::BeginTooltip();
    //   float focus_sz = 32.0f;
    //   float focus_x = ImGui::GetMousePos().x - tex_screen_pos.x - focus_sz * 0.5f; if (focus_x < 0.0f) focus_x = 0.0f; else if (focus_x > tex_w - focus_sz) focus_x = tex_w - focus_sz;
    //   float focus_y = ImGui::GetMousePos().y - tex_screen_pos.y - focus_sz * 0.5f; if (focus_y < 0.0f) focus_y = 0.0f; else if (focus_y > tex_h - focus_sz) focus_y = tex_h - focus_sz;
    //   ImGui::Text("Min: (%.2f, %.2f)", focus_x, focus_y);
    //   ImGui::Text("Max: (%.2f, %.2f)", focus_x + focus_sz, focus_y + focus_sz);
    //   ImVec2 uv0 = ImVec2((focus_x) / tex_w, (focus_y) / tex_h);
    //   ImVec2 uv1 = ImVec2((focus_x + focus_sz) / tex_w, (focus_y + focus_sz) / tex_h);
    //   ImGui::Image((void*)mgpsVars.map_texture.id, ImVec2(256,256), uv0, uv1, ImColor(255,255,255,255), ImColor(255,255,255,128));
    //   ImGui::EndTooltip();
    // }
  }

  ImGui::EndChild();
  ImGui::PopStyleVar();



  static char save_rendered_map_path[256];
  ImGui::PushItemWidth(250);
  ImGui::InputText("###save_rendered_map_path", save_rendered_map_path, 256);
  ImGui::PopItemWidth();
  ImGui::SameLine();

  if (ImGui::Button("save")) {
    printf("saving map screenshot: %s\n", save_rendered_map_path);
    saveGUIRegion((int)mgpsVars.map_texture_info.screen_pos_x+1, (int)mgpsVars.map_texture_info.screen_pos_y,
                  (int)mgpsVars.map_texture_info.width, (int)mgpsVars.map_texture_info.height,
                  save_rendered_map_path);
  }


  ImGui::PushItemWidth(150);
  ImGui::DragInt("overlay alpha", &overlay_transparency, 1.0f, 0, 255);
  ImGui::PopItemWidth();

  ImGui::SameLine();
  ImGui::RadioButton("NN pose", &overlay_idx, 0); ImGui::SameLine();
  ImGui::RadioButton("image pose", &overlay_idx, 1);
  
  ImGui::SameLine();
  ImGui::Checkbox("Camera", &mgpsVars.draw_camera);


  ImGui::End();
  ImGui::PopStyleVar();

}


void drawGui() {
  if (FLAGS_batch_test) {
    drawBatchTestMonitor();
  } else {
    drawSetting();
  }
  drawTestImage();
  drawAlignment();
  drawMap();

}



void batchTestInit() {
  // static bool initialized = false;
  if (batch_test_initialized) {
    return;
  }
  char dataset_path[256];
  sprintf(dataset_path, "%s/%s", mgpsVars.dataset_root, FLAGS_dataset.c_str());
  eventLoadDataset(dataset_path);
  if (!FLAGS_nogui) {
    eventLoadMap(FLAGS_map.c_str(), FLAGS_map_scale);
  } 
  eventInitMicroGPS(FLAGS_cell_size, FLAGS_num_scale_groups, FLAGS_feature_database.c_str(), FLAGS_pca_basis.c_str(), FLAGS_dimensionality);
  strcpy(mgpsVars.test_results_root, FLAGS_output.c_str());
  mkdirIfNotExists(mgpsVars.test_results_root);

  // load test sequence
  mgpsVars.dataset->loadTestSequenceByName(FLAGS_testset.c_str());

  // load first frame
  if (!FLAGS_nogui) {
    WorkImage* current_test_frame = new WorkImage(mgpsVars.dataset->getTestImage(mgpsVars.test_index));
    current_test_frame->loadImage();
    mgpsVars.test_image_texture.loadTextureFromWorkImage(current_test_frame);
    delete current_test_frame;
  }

  // set options
  mgpsVars.options.reset();
  mgpsVars.options.do_alignment = false;
  mgpsVars.options.do_siftmatch = true;
  mgpsVars.options.debug_mode = true;
  mgpsVars.options.image_scale_for_sift = FLAGS_sift_extraction_scale;
  mgpsVars.options.best_knn = FLAGS_best_knn;


  mgpsVars.draw_camera = true;
  // batch_test_initialized = true;
  batch_test_initialized = true;
}


int main(int argc, char *argv[]) {
#ifdef ON_TEGRA
  google::ParseCommandLineFlags(&argc, &argv, true);
#else
  gflags::ParseCommandLineFlags(&argc, &argv, true);
#endif

  mgpsVars.loadDefaultValues();
#ifdef USE_SIFT_GPU
  initSiftGPU();
#endif
  readRobotCameraCalibration(robot_camera_calibration_file_path);

  if (FLAGS_nogui) {
    batchTestInit();
    eventTestAll(true);
    while (true) {
      eventTestAll(false);
    }
    return 0;
  }

  // Setup window
  glfwSetErrorCallback(error_callback);
  if (!glfwInit())
      return 1;
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
  GLFWwindow* window = glfwCreateWindow(1440, 900, "MicroGPS OpenGL3 GUI", NULL, NULL);
  glfwMakeContextCurrent(window);
  gl3wInit();


  // Setup ImGui binding
  ImGui_ImplGlfwGL3_Init(window, true);

  // Load Fonts
  // (there is a default font, this is only if you want to change it. see extra_fonts/README.txt for more details)
  ImGuiIO& io = ImGui::GetIO();
  // io.Fonts->AddFontDefault();
  io.Fonts->AddFontFromFileTTF("../imgui_lib/extra_fonts/Cousine-Regular.ttf", 15.0f);
  // io.Fonts->AddFontFromFileTTF("../imgui_lib/extra_fonts/DroidSans.ttf", 14.0f);
  // io.Fonts->AddFontFromFileTTF("../imgui_lib/extra_fonts/ProggyClean.ttf", 13.0f);
  // io.Fonts->AddFontFromFileTTF("../imgui_lib/extra_fonts/ProggyTiny.ttf", 10.0f);
  // io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesJapanese());

  ImVec4 clear_color = ImColor(114, 144, 154);

  printf("entering glfw loop\n");

  // Main loop
  while (!glfwWindowShouldClose(window))
  {
      glfwPollEvents();
      ImGui_ImplGlfwGL3_NewFrame();

      glfwGetWindowSize(window, &glfwDisplay.screen_w, &glfwDisplay.screen_h);
      glfwGetFramebufferSize(window, &glfwDisplay.framebuffer_w, &glfwDisplay.framebuffer_h);

      drawGui();

      if (FLAGS_batch_test) { // we need to draw one frame before calling init
        batchTestInit();  
      }

      // Rendering
      glViewport(0, 0, glfwDisplay.screen_w, glfwDisplay.screen_h);
      glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      ImGui::Render();
      glfwSwapBuffers(window);
  }

  // Cleanup
  ImGui_ImplGlfwGL3_Shutdown();
  glfwTerminate();

  return 0;
}
