#ifndef _FLYCAPTURE_WRAPPER_H_
#define _FLYCAPTURE_WRAPPER_H_

#include "FlyCapture2.h"
#include <opencv2/opencv.hpp>

class FlyCaptureWrapper {
public:
  FlyCaptureWrapper();
  ~FlyCaptureWrapper();
  void printBuildInfo();
  void printCameraInfo();
  void printCameraProperty(int arrow = -1);
  void readCalibrationFile(const char* calibration_file);

  void setCameraProperty();
  void configureCamera();
  void startCapture();
  void stopCapture();
  unsigned char* getNewFrame();
  int rows();
  int cols();
  int stride();


private:
  void printError(FlyCapture2::Error error);

  float m_shutter_speed;
  float m_view_frame_rate;
  float m_save_frame_rate;
  float m_gain;
  bool m_auto_gain;
  float m_window_resize_ratio;
  int m_sharpness;
  bool m_undistort;

  unsigned int m_n_cols;
  unsigned int m_n_rows;
  unsigned int m_stride;
  FlyCapture2::Camera m_cam;
  FlyCapture2::CameraInfo m_cam_info;
  FlyCapture2::Image m_new_frame;
  cv::Mat undistorted_frame;
  // FlyCapture2::BusManager m_busMgr;
  // FlyCapture2::PGRGuid m_guid;
 
};




#endif
