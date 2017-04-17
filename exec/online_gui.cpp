#include <imgui.h>
#include "imgui_impl_glfw_gl3.h"
#include <GL/gl3w.h>
#include <GLFW/glfw3.h>
#include <fstream>
#include <vector>
#include "dirent.h"
#include <stdio.h>
#include "gui_helper.h"
#include "robot.h"
#include <sys/time.h>
#include <thread>
#include <atomic>
#include <mutex>

// #define WITHOUT_FLYCAPTURE
#ifdef ON_MAC
  #define WITHOUT_FLYCAPTURE
#endif

#define GUI_TEST_WIDTH    (glfwDisplay.screen_h/2 - GUI_GAP_SIZE/2)
// #define GUI_SETTING_WIDTH 200
#define GUI_GAP_SIZE      3

#ifdef ON_MAC
#define GUI_SCALE_FACTOR  2
#endif
#ifdef ON_TEGRA
#define GUI_SCALE_FACTOR  1
#endif

#define GUI_LIVE_HEIGHT 360 * GUI_SCALE_FACTOR
#define GUI_LIVE_WIDTH  200 * GUI_SCALE_FACTOR
#define GUI_MAP_HEIGHT  360 * GUI_SCALE_FACTOR
#define GUI_MAP_WIDTH   240 * GUI_SCALE_FACTOR
#define WORLD_HEIGHT    6000.0f

#define RX_INTERVAL           500
#define TX_INTERVAL           200 * 1000
#define FETCH_IMAGE_INTERVAL  50 * 1000

// file path
#ifdef ON_TEGRA
#include "flycapture_wrapper.h"
FlyCaptureWrapper* flycapture_wrapper;
char dataset_path[] = "usbdrive/micro_gps_packed/cs4_hallway_long_packed";
char feature_database_path[] = "usbdrive/databases/cs4_hallway_siftgpu.bin";
char calibration_file_path[] = "usbdrive/calibration/calibration1111.yml";
char map_image_path[] = "usbdrive/maps/cs4_map_10per.png";
float map_scale = 0.1f;
char robot_camera_calibration_file_path[] = "usbdrive/calibration/robot_camera_calibration.txt";
#endif

#ifdef ON_MAC
char dataset_path[] = "/Users/lgzhang/Documents/DATA/micro_gps_packed/cs4_hallway_long_packed";
char feature_database_path[] = "databases/cs4_hallway_siftgpu.bin";
char calibration_file_path[] = "/Users/lgzhang/micro_gps/code/micro-gps-flycapture2/calibration/calibration1111.yml";
char map_image_path[] = "maps/cs4_map_10per.png";
float map_scale = 0.1f;
char robot_camera_calibration_file_path[] = "robot_camera_calibration.txt";
#endif

Database dataset(dataset_path);
int test_image_idx = 0;
MicroGPS micro_gps;
MicroGPSOptions micro_gps_options;


long long prev_rx_timestamp = 0;
long long prev_tx_timestamp = 0;
long long prev_fetch_image_timestamp = 0;


// related to rendering
Robot robot;
GLFWDisplay glfwDisplay;
WorkImageGL3Texture live_image_texture;
WorkImageGL3Texture map_image_texture;
RenderedTextureInfo map_texture_info;
float world_min_x;
float world_min_y;
int footprint_x[30000];
int footprint_y[30000];
int footprint_angle[30000];
int footprint_idx = -1;;

WorkImage* current_test_frame = NULL;
std::mutex current_test_frame_mutex;

MicroGPSResult micro_gps_result_copy;
std::mutex micro_gps_result_mutex;
MicroGPSTiming micro_gps_timing_copy; 
std::mutex serial_rx_status_mutex;
RobotStatus robot_status_copy;
std::atomic<bool> run_micro_gps_thread(true);
std::atomic<bool> robot_can_track(false);
std::atomic<bool> run_serial_rx_thread(true);
std::atomic<bool> robot_is_initialized(false);

Eigen::Matrix3f T_camera_robot;
float mm_per_pixel = 0;


void readRobotCameraCalibration (char* file_path) {
  T_camera_robot = Eigen::MatrixXf::Identity(3, 3);
  FILE* fp = fopen(file_path, "r");
  fscanf(fp, "%f %f %f\n", &T_camera_robot(0, 0), &T_camera_robot(0, 1), &T_camera_robot(0, 2));
  fscanf(fp, "%f %f %f\n", &T_camera_robot(1, 0), &T_camera_robot(1, 1), &T_camera_robot(1, 2));
  fscanf(fp, "%f\n", &mm_per_pixel);
  fclose(fp);

  std::cout << "T_camera_robot = " << T_camera_robot << std::endl;
  std::cout << "mm_per_pixel = " << mm_per_pixel << std::endl;
}

Eigen::Matrix3f cameraPose2RobotPose(Eigen::Matrix3f T_camera) {
  T_camera(0, 2) *= mm_per_pixel;
  T_camera(1, 2) *= mm_per_pixel;
  return T_camera * T_camera_robot;
}


long long getSystemTime() {
  struct timeval tnow;
  gettimeofday(&tnow, NULL);
  return tnow.tv_sec * 1000000 + tnow.tv_usec;
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



void computeMapOffsets() {
  int n_images = dataset.getDatabaseSize();
  std::vector<WorkImage*> work_images(n_images);
  std::vector<Eigen::Matrix3f> work_image_poses(n_images);

  for (int i = 0; i < n_images; i++) {
    work_images[i] = new WorkImage(dataset.getDatabaseImage(i));
    work_image_poses[i] = dataset.getDatabasePose(i);
  }

  int world_min_x_;
  int world_min_y_;
  int world_max_x_;
  int world_max_y_;

  computeImageArrayWorldSize(work_images,
                            work_image_poses,
                            map_scale,
                            world_min_x_,
                            world_min_y_,
                            world_max_x_,
                            world_max_y_);

  for (int i = 0; i < n_images; i++) {
    work_images[i]->release();
    delete work_images[i];
  }

  world_min_x = world_min_x_;
  world_min_y = world_min_y_;
}

void globalCoordinates2TextureCoordinates(float& x, float& y) {
  x = (x * map_scale - world_min_x) * map_texture_info.fitting_scale;
  y = (y * map_scale - world_min_y) * map_texture_info.fitting_scale;
}

void getNewFrame() {
#ifdef WITHOUT_FLYCAPTURE // we don't have access to the camera :(
  current_test_frame_mutex.lock();
  if (current_test_frame) {
    current_test_frame->release();
    delete current_test_frame;
    current_test_frame = NULL;
  }
  test_image_idx %= dataset.getTestSize();
  current_test_frame = new WorkImage(dataset.getTestImage(test_image_idx++));
  current_test_frame->loadImage();
  current_test_frame_mutex.unlock();
  live_image_texture.loadTextureFromWorkImage(current_test_frame); 
#else
  current_test_frame_mutex.lock();
  if (current_test_frame) {
    current_test_frame->release();
    delete current_test_frame;
    current_test_frame = NULL;
  }
  current_test_frame = new WorkImage(1288, 964, false);
  unsigned char* data = flycapture_wrapper->getNewFrame();
  memcpy(current_test_frame->data(), data, 1288 * 964);
  current_test_frame_mutex.unlock();
  live_image_texture.loadTextureFromWorkImage(current_test_frame); 
    
#endif
}


void MicroGPSThread() {
#ifdef WITHOUT_FLYCAPTURE
  WorkImage current_test_frame_copy(1288, 964);
#else
  WorkImage current_test_frame_copy(1288, 964, false);
#endif  

  MicroGPSResult micro_gps_result;
  MicroGPSTiming micro_gps_timing;  
  MicroGPSDebug micro_gps_debug;
  
  float prev_estimated_x = 9999.0f;
  float prev_estimated_y = 9999.0f;
  
  while (run_micro_gps_thread) {
    WorkImage* alignment_image = NULL;

    bool frame_available = false;
    // do a local copy
    current_test_frame_mutex.lock();
    if (current_test_frame) {
      // printf("current_test_frame_copy: %d x %d\n", current_test_frame_copy.width(), current_test_frame_copy.height());
      if (current_test_frame_copy.data() == NULL) {
        printf("buffer is empty!\n");
      }
      // printf("current_test_frame: %d x %d\n", current_test_frame->width(), current_test_frame->height());      
      memcpy(current_test_frame_copy.data(), current_test_frame->data(), 1288 * 964 * current_test_frame->channels());
      frame_available = true;
    }
    current_test_frame_mutex.unlock();

    if (!frame_available) {
      continue;
    }
    
    printf("start testing\n");
    // micro_gps_result.final_estimated_pose = Eigen::MatrixXf::Identity(3, 3);
    // bool success;
    // current_test_frame_copy.extractSIFT();
    // current_test_frame_copy.PCADimReduction(micro_gps.m_PCA_basis);
    // std::vector<int> nn_index;
    // micro_gps.searchNearestNeighborsMultiScales(&current_test_frame_copy, nn_index);
    
    //TODO: notify robot to track
    if (robot_is_initialized) {
      robot.startTracking();
    }

    bool success = micro_gps.locate(&current_test_frame_copy, alignment_image,
                                    micro_gps_options, micro_gps_result,
                                    micro_gps_timing, micro_gps_debug);
    if (!robot_can_track && success) {
      robot_can_track = true;
      micro_gps_options.do_siftmatch = false;
      prev_estimated_x = micro_gps_result.final_estimated_pose(0, 2);
      prev_estimated_y = micro_gps_result.final_estimated_pose(1, 2);
    } else if (robot_can_track && success) {
      Eigen::Vector2f dist;
      dist(0) = prev_estimated_x - micro_gps_result.final_estimated_pose(0, 2);
      dist(1) = prev_estimated_y - micro_gps_result.final_estimated_pose(1, 2);
      printf("dist = %f\n", dist.norm() * mm_per_pixel);
      if (dist.norm() * mm_per_pixel > 500.0f) { // too far from the last location
        success = false;
        robot_can_track = false;
        micro_gps_options.do_siftmatch = true;
      } else {
        prev_estimated_x = micro_gps_result.final_estimated_pose(0, 2);
        prev_estimated_y = micro_gps_result.final_estimated_pose(1, 2);
      }
    } else {
      robot_can_track = false;
      micro_gps_options.do_siftmatch = true;
    }
    micro_gps_result.success_flag = success;

    // TODO: notify robot to update position
    if (robot_can_track) {
      if (robot_is_initialized) {
        Eigen::Matrix3f robot_pose = cameraPose2RobotPose(micro_gps_result.final_estimated_pose);
        float robot_angle = atan2(-robot_pose(1, 0), robot_pose(0, 0)) / M_PI * 1800.0f;
        if (robot_angle < 0) {
          robot_angle += 3600.0f;
        }
        float robot_x = robot_pose(0, 2); // send in mm
        float robot_y = -robot_pose(1, 2); 
        robot.integrateGlobalPosition((int16_t)robot_x, (int16_t)robot_y, (int16_t)robot_angle);
        printf("sending location %f, %f, %f\n", robot_x, robot_y, robot_angle);
      }
    }

    micro_gps_result_mutex.lock();
    micro_gps_result_copy = micro_gps_result;
    micro_gps_timing_copy = micro_gps_timing;
    micro_gps_result_mutex.unlock();
    // sleep(1);
  }
  printf("micro_gps quitting\n");

}

void SerialRxThread() {
  while (run_serial_rx_thread) {
    RobotStatus& robot_status = robot.getAssembledStatus();
    if (robot_is_initialized) {
      if (robot.readStatusUntilFinish()) {
        serial_rx_status_mutex.lock();
        robot.assembleStatus();
        robot_status.timestamp = getSystemTime();
        serial_rx_status_mutex.unlock();
      }
    }
  }
}



void initGUI () {
#ifdef USE_SIFT_GPU
  initSiftGPU();
#endif


  dataset.loadDatabase();

#ifdef WITHOUT_FLYCAPTURE
  dataset.loadTestSequenceByName("test.test");
#endif
  micro_gps.loadDatabaseOnly(&dataset);

  float cell_size = 50.0f;
  int PCA_dimensions = 16;
  int num_scale_groups = 10;
  micro_gps.setVotingCellSize(cell_size);
  micro_gps.setNumScaleGroups(num_scale_groups);
  micro_gps.loadFeatures(feature_database_path);
  micro_gps.PCAreduction(PCA_dimensions);
  micro_gps.buildSearchIndexMultiScales();

  micro_gps_options.do_alignment = false;
  micro_gps_options.debug_mode = false;
  micro_gps_options.do_siftmatch = true;
  micro_gps_options.confidence_thresh = 0.8f;



  WorkImage new_map(map_image_path);
  new_map.loadImage();
  // bool rotate90 = false;
  // if (mgpsVars.map_texture_avail_height > mgpsVars.map_texture_avail_width != 
  //     new_map->height() > new_map->width()) {
  //   rotate90 = true;
  // }
  map_image_texture.loadTextureFromWorkImage(&new_map, false);
  computeMapOffsets();

#ifndef WITHOUT_FLYCAPTURE
  flycapture_wrapper = new FlyCaptureWrapper();
  flycapture_wrapper->printBuildInfo();
  flycapture_wrapper->readCalibrationFile(calibration_file_path);
  flycapture_wrapper->configureCamera();
  printf("camera configured\n");
  //flycapture_wrapper.printCameraInfo();
  flycapture_wrapper->printCameraProperty();
  flycapture_wrapper->startCapture();
#endif

  readRobotCameraCalibration (robot_camera_calibration_file_path);

}


void drawLive() {
  ImGui::SetNextWindowSize(ImVec2(GUI_LIVE_WIDTH, GUI_LIVE_HEIGHT));
  ImGui::SetNextWindowPos(ImVec2(glfwDisplay.screen_w-GUI_LIVE_WIDTH, 0));

  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(1.0f,1.0f));  

  ImGui::Begin("Live", NULL,  ImGuiWindowFlags_NoCollapse|
                                  ImGuiWindowFlags_NoResize|
                                  ImGuiWindowFlags_NoMove);

  // ImGui::SetNextTreeNodeOpen(true, ImGuiSetCond_Once);
  // if (ImGui::CollapsingHeader("Dataset"))
  // {
  // }

  ImVec2 region_avail = ImGui::GetContentRegionAvail(); // exluding padding
  // ImVec2 tex_screen_pos = ImGui::GetCursorScreenPos();
  float tex_w = (float)ImGui::GetIO().Fonts->TexWidth;
  float tex_h = (float)ImGui::GetIO().Fonts->TexHeight;
  ImTextureID tex_id = ImGui::GetIO().Fonts->TexID;


  if (getSystemTime() - prev_fetch_image_timestamp > FETCH_IMAGE_INTERVAL) {
    prev_fetch_image_timestamp = getSystemTime();
    getNewFrame();
  }
  ImGui::Image((void*)live_image_texture.id, ImVec2(region_avail.x, region_avail.x / 4.0f * 3.0f), ImVec2(0,0), ImVec2(1,1), ImColor(255,255,255,255), ImColor(255,255,255,128));
  


  ImGuiIO& io = ImGui::GetIO();
  bool W_down = ImGui::IsKeyDown('W');
  bool S_down = ImGui::IsKeyDown('S');
  bool A_down = ImGui::IsKeyDown('A');
  bool D_down = ImGui::IsKeyDown('D');
  bool I_pressed = ImGui::IsKeyReleased('I');
  bool P_pressed = ImGui::IsKeyReleased('P');
  bool H_pressed = ImGui::IsKeyReleased('H');
  bool J_pressed = ImGui::IsKeyReleased('J');
  bool K_pressed = ImGui::IsKeyReleased('K');
  bool L_pressed = ImGui::IsKeyReleased('L');  
  bool M_pressed = ImGui::IsKeyReleased('M');
  bool R_pressed = ImGui::IsKeyReleased('R');
  bool T_pressed = ImGui::IsKeyReleased('T');
  bool G_pressed = ImGui::IsKeyReleased('G');

  // bool key_release = ImGui::IsKeyReleased(87) || ImGui::IsKeyReleased(83) || ImGui::IsKeyReleased(65) || ImGui::IsKeyReleased(68);

  ImGui::Text("W:%d S:%d A:%d D:%d P:%d", W_down, S_down, A_down, D_down, P_pressed);
  ImGui::Text("H:%d J:%d K:%d L:%d", H_pressed, J_pressed, K_pressed, L_pressed);
  ImGui::Text("P:dump path, H:send path");
  ImGui::Text("K:follow, M:toggle manual");

  // ImGui::Text("I:%d", I_pressed);


  // Do all the robot things here
  long long now_timestamp = getSystemTime();


  if (I_pressed) {
    if (!robot_is_initialized) {
      robot.initSerial();
      robot_is_initialized = true;
      footprint_idx = -1;
    } else {
      robot_is_initialized = false;
      robot.closeSerial();
    }
  }


  if (P_pressed) { // dump path
    FILE* fp = fopen("path.txt", "w");
    for (int i = 0; i <= footprint_idx; i++) {
      fprintf(fp, "%d %d %d\n", footprint_x[i], footprint_y[i], footprint_angle[i]);      
    }
    fclose(fp);
  }


  if (M_pressed && robot_is_initialized) {
    robot.sendSingleCommand(SET_MANUAL_CMD);
  }


  if (H_pressed && robot_is_initialized) {
    robot.sendSingleCommand(CLEAR_PATH_CMD);
    FILE* fp = fopen("path.txt", "r");
    int16_t path_point_x, path_point_y, path_point_angle;
    int counter = 0;
    while (fscanf(fp, "%hd %hd %hd\n", &path_point_x, &path_point_y, &path_point_angle) != EOF) {
      // printf("%hd %hd %hd\n", path_point_x, path_point_y, path_point_angle);
      robot.setPathPoint(path_point_x, path_point_y, path_point_angle);
      counter++;
    }
    fclose(fp);
    printf("sent %d path points\n", counter);
  }

  if (J_pressed && robot_is_initialized) {
    robot.sendSingleCommand(SET_FOLLOW_PATH_CMD);    
  }

  if (R_pressed && robot_is_initialized) {
    robot.setOdometryPosition(0, 0, 0);
  }

  if (K_pressed) {
    footprint_idx = -1;
  }

  if (T_pressed) {
    robot.startTracking();
  }

  if (G_pressed) {
    robot.integrateGlobalPosition(0, 0, 0);
  }

  // if (now_timestamp - prev_tx_timestamp > TX_INTERVAL) {
  //   prev_tx_timestamp = now_timestamp;
  //   if (robot_is_initialized && robot_status_copy.manual_mode) {
  //     int8_t forward_speed = 0;
  //     int8_t rotate_speed = 0;
  //     if (W_down) {
  //       forward_speed += 100;
  //     }
  //     if (S_down) {
  //       forward_speed -= 100;
  //     }
  //     if (A_down) {
  //       rotate_speed += 30;
  //     }
  //     if (D_down) {
  //       rotate_speed -= 30;
  //     }
  //     robot.setSpeed(forward_speed, rotate_speed);
  //   }
  // }

  // RobotStatus& robot_status = robot.getAssembledStatus();
  // printf("now_timestamp = %lld, prev_rx_timestamp = %lld, diff = %lld\n", now_timestamp, prev_rx_timestamp, now_timestamp - prev_rx_timestamp);
  // if (now_timestamp - prev_rx_timestamp > RX_INTERVAL) {
  //   prev_rx_timestamp = now_timestamp;
  //   if (robot_is_initialized) {
  //     printf("trying to read status\n");
  //     if (robot.readStatus()) {
  //       printf("successfully got status\n");
  //       robot.assembleStatus();
  //       robot_status.timestamp = now_timestamp;
  //     }
  //   }
  // }


  ImGui::Text("%s", robot_is_initialized?"Connected":"Disconnected(press I)");

  ImGui::Text("Now=%lld", now_timestamp);

  // ImGui::Text("t=%lld\nX=%fmm Y=%fmm A=%.01f ", robot_status_copy.timestamp,
  //                                               (float)robot_status_copy.x * 10.0f, 
  //                                               (float)robot_status_copy.y * 10.0f, 
  //                                               (float)robot_status_copy.angle / 10.0f);

  ImGui::Text("t=%lld\nX=%.0fmm Y=%.0fmm A=%.01f ", robot_status_copy.timestamp,
                                                (float)robot_status_copy.x, 
                                                (float)robot_status_copy.y, 
                                                (float)robot_status_copy.angle / 10.0f);

  micro_gps_result_mutex.lock();
  ImGui::Text("Test result: %s", micro_gps_result_copy.success_flag ? "SUCCESS" : "FAILURE");
  ImGui::Text("Total time: %.3lf ms", micro_gps_timing_copy.total * 1000.0f);
  ImGui::BulletText("SIFT extraction : %.3lf ms", micro_gps_timing_copy.sift_extraction * 1000.0f);
  ImGui::BulletText("KNN search : %.3lf ms", micro_gps_timing_copy.knn_search * 1000.0f);
  // ImGui::BulletText("Compute candidate poses : %.3lf ms", micro_gps_timing_copy.candidate_image_pose * 1000.0f);
  // ImGui::BulletText("Voting : %.3lf ms", micro_gps_timing_copy.voting * 1000.0f);
  // ImGui::BulletText("RANSAC : %.3lf ms", micro_gps_timing_copy.ransac * 1000.0f);
  if (micro_gps_result_copy.top_cells.size() > 0) {
    int hist_max = *max_element(micro_gps_result_copy.top_cells.begin(), micro_gps_result_copy.top_cells.end());
    int n_cells = micro_gps_result_copy.top_cells.size();
    float* data = new float[n_cells];
    for (int i = 0; i < n_cells; i++) {
      data[i] = (float)micro_gps_result_copy.top_cells[i];
    }
    // ImGui::PlotHistogram("###voting_hist", data,
    //           n_cells, 0, "hello", 0.0f, (float)hist_max, ImVec2(size.x, 50));

    // ImGui::Text("Top cells");
    ImGui::Columns(n_cells, NULL, true);
    ImGui::Separator();
    for (int i = 0; i < n_cells; i++) {
        ImGui::Text("%.0f", data[i]);
        ImGui::NextColumn();
    }
    ImGui::Columns(1);
    ImGui::Separator();
    delete[] data;
  }



  micro_gps_result_mutex.unlock();

  ImGui::End();
  ImGui::PopStyleVar();
  ImGui::PopStyleVar();

}

// void drawTestImage() {
//   ImGui::SetNextWindowSize(ImVec2(GUI_TEST_WIDTH, GUI_TEST_WIDTH));
//   ImGui::SetNextWindowPos(ImVec2(glfwDisplay.screen_w-GUI_TEST_WIDTH-GUI_SETTING_WIDTH-GUI_GAP_SIZE, 0));
//   ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
//   ImGui::Begin("Test Image", NULL,  ImGuiWindowFlags_NoCollapse|
//                                     ImGuiWindowFlags_NoResize|
//                                     ImGuiWindowFlags_NoMove|
//                                     ImGuiWindowFlags_NoScrollbar);


//   ImVec2 region_avail = ImGui::GetContentRegionAvail(); // exluding padding
//   // region_avail.y -= ImGui::GetItemsLineHeightWithSpacing();  // reserving space for other widgets

//   // ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(1.0f, 1.0f));
//   // ImGui::BeginChild("test_image_disp", ImVec2(region_avail.x, region_avail.y), true);

//   // ImVec2 tex_screen_pos;
//   // float tex_w;
//   // float tex_h;

//   // // display the image
//   // if (mgpsVars.test_image_texture.show) {
//   //   // compute texture size
//   //   ImVec2 im_disp_region = ImGui::GetContentRegionAvail();
//   //   float im_scaling = std::min(im_disp_region.x / mgpsVars.test_image_texture.width, im_disp_region.y / mgpsVars.test_image_texture.height);
//   //   tex_w = mgpsVars.test_image_texture.width * im_scaling;
//   //   tex_h = mgpsVars.test_image_texture.height * im_scaling;

//   //   if (im_disp_region.x > tex_w) {
//   //     ImGui::Indent((im_disp_region.x - tex_w) / 2.0f);
//   //   }

//   //   tex_screen_pos = ImGui::GetCursorScreenPos();
//   //   // ImGui::Image((void*)mgpsVars.test_image_texture.id, ImVec2(tex_w, tex_h), ImVec2(0,0), ImVec2(1,1), ImColor(255,255,255,255), ImColor(0,0,0,0));


//   //   ImGui::GetWindowDrawList()->AddImage((void*)mgpsVars.test_image_texture.id, ImVec2(tex_screen_pos.x, tex_screen_pos.y),
//   //                                       ImVec2(tex_screen_pos.x+tex_w, tex_screen_pos.y+tex_h));

//   // }

//   // ImGui::EndChild();
//   // ImGui::PopStyleVar();


//   ImGui::End();
//   ImGui::PopStyleVar();
// }

void drawMap() {
  ImGui::SetNextWindowSize(ImVec2(GUI_MAP_WIDTH, GUI_MAP_HEIGHT));
  ImGui::SetNextWindowPos(ImVec2(0, 0));

  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(1.0f,1.0f));  

  ImGui::Begin("Map", NULL,  ImGuiWindowFlags_NoCollapse|
                                  ImGuiWindowFlags_NoResize|
                                  ImGuiWindowFlags_NoMove);

  ImVec2 topleft_screen_pos = ImGui::GetCursorScreenPos();
  ImVec2 region_avail = ImGui::GetContentRegionAvail();
  ImGui::GetWindowDrawList()->AddLine(ImVec2(0, topleft_screen_pos.y + region_avail.y/2.0f), 
                                      ImVec2(topleft_screen_pos.x + region_avail.x, topleft_screen_pos.y + region_avail.y/2.0f),
                                      ImColor(255,0,0,255), 1.0f);

  ImGui::GetWindowDrawList()->AddLine(ImVec2(topleft_screen_pos.x + region_avail.x/2.0f, 0), 
                                      ImVec2(topleft_screen_pos.x + region_avail.x/2.0f, topleft_screen_pos.y + region_avail.y),
                                      ImColor(0,255,0,255), 1.0f);


  // draw map
  map_texture_info.fitting_scale = std::min(region_avail.x / map_image_texture.getWidth(), region_avail.y / map_image_texture.getHeight());
  float tex_w = map_image_texture.getWidth() * map_texture_info.fitting_scale;
  float tex_h = map_image_texture.getHeight() * map_texture_info.fitting_scale;

  if (region_avail.x > tex_w) {
    ImGui::Indent((region_avail.x - tex_w) / 2.0f);
  }
  ImVec2 tex_screen_pos = ImGui::GetCursorScreenPos();

  // update texture info
  map_texture_info.width = tex_w;
  map_texture_info.height = tex_h;
  map_texture_info.screen_pos_x = tex_screen_pos.x;
  map_texture_info.screen_pos_y = tex_screen_pos.y;

  ImGui::Image((void*)map_image_texture.id, ImVec2(tex_w, tex_h), ImVec2(0,0), ImVec2(1,1), ImColor(255,255,255,255), ImColor(0,0,0,0));
  ImGui::SetCursorScreenPos(tex_screen_pos); // go back
  
  micro_gps_result_mutex.lock();  
  // float center_x = micro_gps_result_copy.final_estimated_pose(0, 2);
  // float center_y = micro_gps_result_copy.final_estimated_pose(1, 2);
  // float x_axis_x = micro_gps_result_copy.final_estimated_pose(0, 0);
  // float x_axis_y = micro_gps_result_copy.final_estimated_pose(1, 0);
  // float y_axis_x = micro_gps_result_copy.final_estimated_pose(0, 1);
  // float y_axis_y = micro_gps_result_copy.final_estimated_pose(1, 1);

  Eigen::Matrix3f robot_pose = cameraPose2RobotPose(micro_gps_result_copy.final_estimated_pose);
  // std::cout << "camera_pose = \n" << micro_gps_result_copy.final_estimated_pose << std::endl;
  // std::cout << "robot_pose = \n" << robot_pose << std::endl;

  micro_gps_result_mutex.unlock();  

  float center_x = robot_pose(0, 2) / mm_per_pixel;
  float center_y = robot_pose(1, 2) / mm_per_pixel;
  float x_axis_x = robot_pose(0, 0);
  float x_axis_y = robot_pose(1, 0);
  float y_axis_x = robot_pose(0, 1);
  float y_axis_y = robot_pose(1, 1);

  globalCoordinates2TextureCoordinates(center_x, center_y);
  center_x += tex_screen_pos.x;
  center_y += tex_screen_pos.y;

  ImGui::GetWindowDrawList()->AddLine(ImVec2(center_x, center_y), ImVec2(center_x + x_axis_x*10*GUI_SCALE_FACTOR, center_y + x_axis_y*10*GUI_SCALE_FACTOR),
                                        ImColor(0,0,255,255), 2.0f);
  ImGui::GetWindowDrawList()->AddLine(ImVec2(center_x, center_y), ImVec2(center_x + y_axis_x*10*GUI_SCALE_FACTOR, center_y + y_axis_y*10*GUI_SCALE_FACTOR),
                                        ImColor(255,0,0,255), 2.0f);
  ImGui::GetWindowDrawList()->AddCircle(ImVec2(center_x, center_y), 10*GUI_SCALE_FACTOR, ImColor(0,255,0,255), 12, 2.0f);

  // get robot location
  // robot.getAssembledStatus().x = 100;
  // robot.getAssembledStatus().y = 200;
  // robot.getAssembledStatus().angle = 200;

  if (robot_is_initialized && robot_status_copy.timestamp > 0) {
    int current_x = robot_status_copy.x;
    int current_y = robot_status_copy.y;
    int current_angle = robot_status_copy.angle;

    if (footprint_idx < 0 || 
        (std::abs(footprint_x[footprint_idx] - current_x) > 100 || 
        std::abs(footprint_y[footprint_idx] - current_y) > 100 ||
        std::abs(footprint_angle[footprint_idx] - current_angle) > 100)) {
      footprint_x[++footprint_idx] = current_x;
      footprint_y[footprint_idx] = current_y;
      footprint_angle[footprint_idx] = current_angle;
    }

    // float n_pixels_per_mm = region_avail.y / WORLD_HEIGHT;
    // float current_x_screen = topleft_screen_pos.x + region_avail.x / 2.0f + (float)current_x * n_pixels_per_mm;
    // float current_y_screen = topleft_screen_pos.y + region_avail.y / 2.0f - (float)current_y * n_pixels_per_mm;
    // float current_angle_radian = (float)current_angle / 1800.0f * M_PI;
    // float x_axis_x = cos(current_angle_radian); 
    // float x_axis_y = -sin(current_angle_radian);
    // float y_axis_x = -sin(current_angle_radian);
    // float y_axis_y = -cos(current_angle_radian);

    // // draw footprints
    // for (int i = 0; i < footprint_idx; i++) {  
    //   float footprint_x_screen = topleft_screen_pos.x + region_avail.x / 2.0f + (float)footprint_x[i] * n_pixels_per_mm;
    //   float footprint_y_screen = topleft_screen_pos.y + region_avail.y / 2.0f - (float)footprint_y[i] * n_pixels_per_mm;
    //   ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(footprint_x_screen, footprint_y_screen), 3, ImColor(0,0,255,255), 12);    
    // }

    // // draw robot
    // ImGui::GetWindowDrawList()->AddLine(ImVec2(current_x_screen - x_axis_x * 20.0f, current_y_screen - x_axis_y * 20.0f), 
    //                                     ImVec2(current_x_screen + x_axis_x * 20.0f, current_y_screen + x_axis_y * 20.0f),
    //                                     ImColor(255,0,0,255), 3.0f);

    // ImGui::GetWindowDrawList()->AddLine(ImVec2(current_x_screen, current_y_screen), 
    //                                     ImVec2(current_x_screen + y_axis_x * 30.0f, current_y_screen + y_axis_y * 30.0f),
    //                                     ImColor(0,255,0,255), 3.0f);


    float center_x = (float)current_x / mm_per_pixel;
    float center_y = -(float)current_y / mm_per_pixel;
    float current_angle_radian = (float)current_angle / 1800.0f * M_PI;
    float x_axis_x = cos(current_angle_radian);
    float x_axis_y = -sin(current_angle_radian);
    float y_axis_x = -sin(current_angle_radian);
    float y_axis_y = -cos(current_angle_radian);

    globalCoordinates2TextureCoordinates(center_x, center_y);
    center_x += tex_screen_pos.x;
    center_y += tex_screen_pos.y;

    // draw footprints
    // for (int i = 0; i < footprint_idx; i++) {  
    //   float footprint_x_screen = topleft_screen_pos.x + region_avail.x / 2.0f + (float)footprint_x[i] * n_pixels_per_mm;
    //   float footprint_y_screen = topleft_screen_pos.y + region_avail.y / 2.0f - (float)footprint_y[i] * n_pixels_per_mm;
    //   ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(footprint_x_screen, footprint_y_screen), 3, ImColor(0,0,255,255), 12);    
    // }

    // draw robot
    ImGui::GetWindowDrawList()->AddLine(ImVec2(center_x, center_y), ImVec2(center_x + x_axis_x*10*GUI_SCALE_FACTOR, center_y + x_axis_y*10*GUI_SCALE_FACTOR),
                                          ImColor(0,0,255,255), 2.0f);
    ImGui::GetWindowDrawList()->AddLine(ImVec2(center_x, center_y), ImVec2(center_x + y_axis_x*10*GUI_SCALE_FACTOR, center_y + y_axis_y*10*GUI_SCALE_FACTOR),
                                          ImColor(255,0,0,255), 2.0f);
  }

  ImGui::End();
  ImGui::PopStyleVar();
  ImGui::PopStyleVar();
}

void drawGui() {
  serial_rx_status_mutex.lock();
  robot_status_copy = robot.getAssembledStatus();
  serial_rx_status_mutex.unlock();

  drawLive();
  drawMap();
  // drawTestImage();
}




int main(int argc, char const *argv[]) {
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
  GLFWwindow* window = glfwCreateWindow(GUI_LIVE_WIDTH + GUI_MAP_WIDTH, 
                                        GUI_LIVE_HEIGHT, 
                                        "MicroGPS OpenGL3 GUI", NULL, NULL);
  glfwMakeContextCurrent(window);
  gl3wInit();


  // Setup ImGui binding
  ImGui_ImplGlfwGL3_Init(window, true);

  // Load Fonts
  // (there is a default font, this is only if you want to change it. see extra_fonts/README.txt for more details)
  ImGuiIO& io = ImGui::GetIO();
  // io.Fonts->AddFontDefault();
  io.Fonts->AddFontFromFileTTF("../imgui_lib/extra_fonts/Cousine-Regular.ttf", 10.0f * GUI_SCALE_FACTOR);
  // io.Fonts->AddFontFromFileTTF("../imgui_lib/extra_fonts/DroidSans.ttf", 14.0f);
  //io.Fonts->AddFontFromFileTTF("../imgui_lib/extra_fonts/ProggyClean.ttf", 13.0f);
  //io.Fonts->AddFontFromFileTTF("../imgui_lib/extra_fonts/ProggyTiny.ttf", 10.0f);
  //io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesJapanese());

  ImVec4 clear_color = ImColor(114, 144, 154);

  // WorkImage* work_image = new WorkImage("im.png");
  // work_image->loadImage();

  // mgpsVars.map_texture.loadTextureFromWorkImage(work_image);
  // mgpsVars.test_image_texture.loadTextureFromWorkImage(work_image);
  // mgpsVars.alignment_texture.loadTextureFromWorkImage(work_image);

  // delete work_image;

  initGUI();
  std::thread micro_gps_thread(MicroGPSThread);
  std::thread serial_rx_thread(SerialRxThread);

  // Main loop
  while (!glfwWindowShouldClose(window))
  {
      glfwPollEvents();
      ImGui_ImplGlfwGL3_NewFrame();

      glfwGetWindowSize(window, &glfwDisplay.screen_w, &glfwDisplay.screen_h);
      glfwGetFramebufferSize(window, &glfwDisplay.framebuffer_w, &glfwDisplay.framebuffer_h);

      drawGui();

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

  printf("glfw window closed\n");

#ifndef WITHOUT_FLYCAPTURE
  flycapture_wrapper->stopCapture();
  delete flycapture_wrapper;
#endif
  run_micro_gps_thread = false;
  run_serial_rx_thread = false;
  micro_gps_thread.join();
  serial_rx_thread.join();

  return 0;
}

