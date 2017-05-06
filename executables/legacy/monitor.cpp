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

#define RX_INTERVAL           2500
#define TX_INTERVAL           20 * 1000

// file path
char dataset_path[] = "/Users/lgzhang/Documents/DATA/micro_gps_packed/cs4_hallway_long_packed";
char map_image_path[] = "maps/cs4_map_10per.png";
float map_scale = 0.1f;
char robot_camera_calibration_file_path[] = "robot_camera_calibration.txt";

long long prev_rx_timestamp = 0;
long long prev_tx_timestamp = 0;

// related to rendering
Robot robot;
GLFWDisplay glfwDisplay;
WorkImageGL3Texture map_image_texture;
RenderedTextureInfo map_texture_info;
float world_min_x;
float world_min_y;
float world_max_x;
float world_max_y;
int footprint_x[30000];
int footprint_y[30000];
int footprint_angle[30000];
int footprint_idx = -1;;
bool record_footprints = false;

Database dataset(dataset_path);
std::mutex serial_rx_status_mutex;
RobotStatus robot_status_copy;
std::atomic<bool> run_serial_rx_thread(true);
std::atomic<bool> robot_is_initialized(false);

Eigen::Matrix3f T_camera_robot;
float mm_per_pixel = 0;

bool ready_to_mark_dest = true;
float marked_dest_x;
float marked_dest_y;
float marked_dest_angle;
float marked_dest_angle_radian;

int forward_speed_val = 50;
int rotate_speed_val = 15;


static void error_callback(int error, const char* description) {
  fprintf(stderr, "Error %d: %s\n", error, description);
}

int angleDiff (int target, int current) {
  int diff = target - current;
  if (diff > 1800) {
    diff -= 3600;
  } else if (diff < -1800) {
    diff += 3600;
  }
  return diff;
}

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

long long getSystemTime() {
  struct timeval tnow;
  gettimeofday(&tnow, NULL);
  return tnow.tv_sec * 1000000 + tnow.tv_usec;
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
  world_max_x = world_max_x_;
  world_max_y = world_max_y_;
}

void globalCoordinates2TextureCoordinates(float& x, float& y) {
  x = (x * map_scale - world_min_x) * map_texture_info.fitting_scale;
  y = (y * map_scale - world_min_y) * map_texture_info.fitting_scale;
}

void textureCoordinates2GlobalCoordinates(float&x , float& y) {
  x = (x / map_texture_info.fitting_scale + world_min_x) / map_scale;
  y = (y / map_texture_info.fitting_scale + world_min_y) / map_scale;
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
  dataset.loadDatabase();
  WorkImage new_map(map_image_path);
  new_map.loadImage();
  map_image_texture.loadTextureFromWorkImage(&new_map, false);
  computeMapOffsets();
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
  bool E_pressed = ImGui::IsKeyReleased('E');

  ImGui::Text("W:%d S:%d A:%d D:%d P:%d", W_down, S_down, A_down, D_down, P_pressed);
  ImGui::Text("H:%d J:%d K:%d L:%d", H_pressed, J_pressed, K_pressed, L_pressed);

  ImGui::Text("forward: %d, rotate: %d", forward_speed_val, rotate_speed_val);
  if (ImGui::IsKeyReleased('-')) {
    forward_speed_val -= 5;
  } else if (ImGui::IsKeyReleased('=')) {
    forward_speed_val += 5;
  }

  if (ImGui::IsKeyReleased('[')) {
    rotate_speed_val -= 5;
  } else if (ImGui::IsKeyReleased(']')) {
    rotate_speed_val += 5;
  }

  // ImGui::Text("P:dump path, H:send path");
  // ImGui::Text("K:follow, M:toggle manual");

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
    fprintf(fp, "%d %d %d\n", robot_status_copy.x, robot_status_copy.y, robot_status_copy.angle);      
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
    robot.setOdometryPosition((int16_t)(world_max_x + world_min_x) / map_scale / 2.0f * mm_per_pixel, 
                              (int16_t)(-(world_max_y + world_min_y) / map_scale / 2.0f * mm_per_pixel), 
                              0.0f);
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

  if (E_pressed) {
    record_footprints ^= 1;
  }

  // if (ImGui::Button("set destination") && robot_is_initialized) {
  //   robot.sendSingleCommand(CLEAR_PATH_CMD);
  //   printf("marked_dest_x = %fmm, marked_dest_y = %fmm, marked_dest_angle = %f\n", marked_dest_x, marked_dest_y, marked_dest_angle);
  //   robot.setPathPoint((int16_t)(marked_dest_x), (int16_t)(marked_dest_y), (int16_t)marked_dest_angle);
  // }


  static int8_t last_forward_speed = 0;
  static int8_t last_rotate_speed = 0;
  if (now_timestamp - prev_tx_timestamp > TX_INTERVAL) {
    prev_tx_timestamp = now_timestamp;
    if (robot_is_initialized && robot_status_copy.manual_mode) {
      int8_t forward_speed = 0;
      int8_t rotate_speed = 0;
      if (W_down) {
        forward_speed += forward_speed_val;
      }
      if (S_down) {
        forward_speed -= forward_speed_val;
      }
      if (A_down) {
        rotate_speed += rotate_speed_val;
      }
      if (D_down) {
        rotate_speed -= rotate_speed_val;
      }
      if (forward_speed != last_forward_speed || rotate_speed != last_rotate_speed) {
        robot.setSpeed(forward_speed, rotate_speed);
        last_forward_speed = forward_speed;
        last_rotate_speed = rotate_speed;
      }
    }
  }

  // RobotStatus& robot_status = robot.getAssembledStatus();
  // printf("now_timestamp = %lld, prev_rx_timestamp = %lld, diff = %lld\n", now_timestamp, prev_rx_timestamp, now_timestamp - prev_rx_timestamp);
  // if (now_timestamp - prev_rx_timestamp > RX_INTERVAL) {
  //   prev_rx_timestamp = now_timestamp;
  //   if (robot_is_initialized) {
  //     // printf("trying to read status\n");
  //     if (robot.readStatusUntilFinish()) {
  //       // printf("successfully got status\n");
  //       robot.assembleStatus();
  //       robot_status.timestamp = now_timestamp;
  //     }
  //   }
  // }


  ImGui::Text("%s", robot_is_initialized?"Connected":"Disconnected(press I)");

  ImGui::Text("Now=%lld", now_timestamp);

  ImGui::Text("t=%lld\nX=%.0fmm Y=%.0fmm A=%.01f ", robot_status_copy.timestamp,
                                                (float)robot_status_copy.x, 
                                                (float)robot_status_copy.y, 
                                                (float)robot_status_copy.angle / 10.0f);

  ImGui::End();
  ImGui::PopStyleVar();
  ImGui::PopStyleVar();

}

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
  

  if (robot_is_initialized && robot_status_copy.timestamp > 0) {
    int current_x = robot_status_copy.x;
    int current_y = robot_status_copy.y;
    int current_angle = robot_status_copy.angle;

    if ((footprint_idx < 0 || 
        (std::abs(footprint_x[footprint_idx] - current_x) > 300 || 
        std::abs(footprint_y[footprint_idx] - current_y) > 300 ||
        std::abs(angleDiff(footprint_angle[footprint_idx], current_angle)) > 200))
        && record_footprints) {
      footprint_idx++;
      footprint_x[footprint_idx] = current_x;
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
    for (int i = 0; i <= footprint_idx; i++) {  
      float footprint_x_screen = (float)footprint_x[i] / mm_per_pixel;
      float footprint_y_screen = -(float)footprint_y[i]/ mm_per_pixel;
      globalCoordinates2TextureCoordinates(footprint_x_screen, footprint_y_screen);
      footprint_x_screen += tex_screen_pos.x;
      footprint_y_screen += tex_screen_pos.y;
      ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(footprint_x_screen, footprint_y_screen), 3, ImColor(0,0,255,255), 12);    
    }


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

  static float mousedown_x;
  static float mousedown_y;
  if (ImGui::IsMouseClicked(0)) {
    if (ImGui::GetIO().MousePos.x < tex_screen_pos.x + map_texture_info.width) {
      mousedown_x = ImGui::GetIO().MousePos.x;
      mousedown_y = ImGui::GetIO().MousePos.y;
      printf("mousedown_x = %f, mousedown_y = %f\n", mousedown_x, mousedown_y);
    }
  }

  if (ImGui::IsMouseDragging()) {
    float mousedragdelta_x = ImGui::GetMouseDragDelta(0).x;
    float mousedragdelta_y = ImGui::GetMouseDragDelta(0).y;
    ImGui::GetWindowDrawList()->AddLine(ImVec2(mousedown_x, mousedown_y), ImVec2(mousedown_x + mousedragdelta_x, mousedown_y + mousedragdelta_y),
                                          ImColor(0,255,0,255), 2.0f);
    
    if (ready_to_mark_dest) {
      marked_dest_x = mousedown_x - tex_screen_pos.x;
      marked_dest_y = mousedown_y - tex_screen_pos.y;
      textureCoordinates2GlobalCoordinates(marked_dest_x, marked_dest_y);
      marked_dest_x = marked_dest_x * mm_per_pixel;
      marked_dest_y = -marked_dest_y * mm_per_pixel;

      marked_dest_angle_radian = atan2(-mousedragdelta_x, -mousedragdelta_y);
      marked_dest_angle = marked_dest_angle_radian / M_PI * 1800.0f;
      if (marked_dest_angle < 0) {
        marked_dest_angle += 3600.0f;
      }
    }
  }
  
  // float x_axis_x = cos(marked_dest_angle_radian);
  // float x_axis_y = -sin(marked_dest_angle_radian);
  // float y_axis_x = -sin(marked_dest_angle_radian);
  // float y_axis_y = -cos(marked_dest_angle_radian);      

  // ImGui::GetWindowDrawList()->AddLine(ImVec2(mousedown_x, mousedown_y), ImVec2(mousedown_x + x_axis_x*10*GUI_SCALE_FACTOR, mousedown_y + x_axis_y*10*GUI_SCALE_FACTOR),
  //                                       ImColor(0,0,255,255), 2.0f);
  // ImGui::GetWindowDrawList()->AddLine(ImVec2(mousedown_x, mousedown_y), ImVec2(mousedown_x + y_axis_x*10*GUI_SCALE_FACTOR, mousedown_y + y_axis_y*10*GUI_SCALE_FACTOR),
  //                                       ImColor(255,0,0,255), 2.0f);
  // ImGui::GetWindowDrawList()->AddCircle(ImVec2(mousedown_x, mousedown_y), 10*GUI_SCALE_FACTOR, ImColor(0,255,0,255), 12, 2.0f);


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


  initGUI();
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


  run_serial_rx_thread = false;
  serial_rx_thread.join();

  return 0;
}

