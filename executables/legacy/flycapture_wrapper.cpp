#include "flycapture_wrapper.h"
//#include "stdafx.h"
#include <iostream>
#include <sstream>
#include "stdio.h"
#include "stdlib.h"
#include "time.h"
//#include <opencv2/opencv.hpp>
#include "calibrate_intrinsics.h"

FlyCaptureWrapper::FlyCaptureWrapper():
m_shutter_speed(3.00f),
m_view_frame_rate(30.00f),
m_save_frame_rate(5.00f),
m_gain(6.96f),
m_auto_gain(true),
m_window_resize_ratio(0.55f),
m_sharpness(1200),
m_undistort(true)
{

  unsigned int numCameras;
  FlyCapture2::BusManager busMgr;

  printError(busMgr.GetNumOfCameras(&numCameras));
  printf("Number of cameras detected: %d\n", numCameras);

  FlyCapture2::PGRGuid guid;
  printError(busMgr.GetCameraFromIndex(0, &guid));

  printError(m_cam.Connect(&guid));
  printf("Camera connected\n");
}

FlyCaptureWrapper::~FlyCaptureWrapper() {
  printError(m_cam.Disconnect());
  printf("Camera disconnected\n");
}


void FlyCaptureWrapper::printError(FlyCapture2::Error error) {
  if (error != FlyCapture2::PGRERROR_OK) {
    error.PrintErrorTrace();
    exit(-1);
  }
}


void FlyCaptureWrapper::printBuildInfo() {
  FlyCapture2::FC2Version fc2Version;
  FlyCapture2::Utilities::GetLibraryVersion(&fc2Version);

  printf("FlyCapture2 library version: %d.%d.%d.%d\n", fc2Version.major,
         fc2Version.minor, fc2Version.type, fc2Version.build);
  printf("Application build date: %s %s\n", __DATE__, __TIME__);
}


void FlyCaptureWrapper::printCameraProperty(int arrow) {
  FlyCapture2::Property prop;
  
  prop.type = FlyCapture2::SHUTTER;
  printError(m_cam.GetProperty(&prop));  
  printf("%sSHUTTER:        Auto = %s     %s    Value = %.02f\n", arrow==0?">":" ", prop.autoManualMode?"True ":"False", prop.onOff?"On ":"OFF", prop.absValue);

  prop.type = FlyCapture2::FRAME_RATE;
  printError(m_cam.GetProperty(&prop));  
  printf("%sFRAME_RATE:     Auto = %s     %s    Value = %.02f\n", arrow==1?">":" ", prop.autoManualMode?"True ":"False", prop.onOff?"On ":"OFF", prop.absValue);

  prop.type = FlyCapture2::SHARPNESS;
  printError(m_cam.GetProperty(&prop));  
  printf("%sSHARPNESS:      Auto = %s     %s    Value = %d   \n", arrow==2?">":" ", prop.autoManualMode?"True ":"False", prop.onOff?"On ":"OFF", prop.valueA);

  prop.type = FlyCapture2::GAIN;
  printError(m_cam.GetProperty(&prop));  
  printf("%sGAIN:           Auto = %s     %s    Value = %.02f\n", arrow==3?">":" ", prop.autoManualMode?"True ":"False", prop.onOff?"On ":"OFF", prop.absValue);

  prop.type = FlyCapture2::AUTO_EXPOSURE;
  printError(m_cam.GetProperty(&prop));  
  printf("%sAUTO_EXPOSURE:  Auto = %s     %s    Value = %.02f\n", arrow==4?">":" ", prop.autoManualMode?"True ":"False", prop.onOff?"On ":"OFF", prop.absValue);

  prop.type = FlyCapture2::GAMMA;
  printError(m_cam.GetProperty(&prop));  
  printf("%sGAMMA:          Auto = %s     %s    Value = %.02f\n", arrow==5?">":" ", prop.autoManualMode?"True ":"False", prop.onOff?"On ":"OFF", prop.absValue);

  prop.type = FlyCapture2::BRIGHTNESS;
  printError(m_cam.GetProperty(&prop));  
  printf("%sBRIGHTNESS:     Auto = %s     %s    Value = %.02f\n", arrow==6?">":" ", prop.autoManualMode?"True ":"False", prop.onOff?"On ":"OFF", prop.absValue);
  
  printf("\n");
}

void FlyCaptureWrapper::setCameraProperty() {
  FlyCapture2::Property prop;

  prop.type = FlyCapture2::SHUTTER;
  prop.absControl = true;
  prop.absValue = m_shutter_speed;
  prop.autoManualMode = false;
  prop.onePush = false;
  prop.onOff = true;
  printError(m_cam.SetProperty(&prop));


  prop.type = FlyCapture2::FRAME_RATE;
  prop.absControl = true;
  prop.absValue = m_view_frame_rate;
  prop.autoManualMode = false;
  prop.onePush = false;
  prop.onOff = true;
  printError(m_cam.SetProperty(&prop));

  prop.type = FlyCapture2::SHARPNESS;
  prop.absControl = false;
  prop.valueA = m_sharpness;
  prop.autoManualMode = false;
  prop.onePush = false;
  prop.onOff = true;
  printError(m_cam.SetProperty(&prop));

  if (!m_auto_gain) {
    prop.type = FlyCapture2::GAIN;
    prop.absControl = true;
    prop.absValue = m_gain;
    prop.autoManualMode = false;
    prop.onePush = false;
    prop.onOff = true;
    printError(m_cam.SetProperty(&prop));

    prop.type = FlyCapture2::AUTO_EXPOSURE;
    prop.absControl = true;
    prop.absValue = 1.00f;
    prop.autoManualMode = false;
    prop.onePush = false;
    prop.onOff = false; //disable
    printError(m_cam.SetProperty(&prop));
  } else {
    prop.type = FlyCapture2::GAIN;
    prop.absControl = true;
    prop.absValue = m_gain;
    prop.autoManualMode = true;
    prop.onePush = false;
    prop.onOff = true;
    printError(m_cam.SetProperty(&prop));

    prop.type = FlyCapture2::AUTO_EXPOSURE;
    prop.absControl = true;
    prop.absValue = 1.00f;
    prop.autoManualMode = true;
    prop.onePush = false;
    prop.onOff = true;
    printError(m_cam.SetProperty(&prop));
  }

}

void FlyCaptureWrapper::printCameraInfo() {
  printError(m_cam.GetCameraInfo(&m_cam_info));
  std::cout << std::endl;
  std::cout << "*** CAMERA INFORMATION ***" << std::endl;
  std::cout << "Serial number -" << m_cam_info.serialNumber << std::endl;
  std::cout << "Camera model - " << m_cam_info.modelName << std::endl;
  std::cout << "Camera vendor - " << m_cam_info.vendorName << std::endl;
  std::cout << "Sensor - " << m_cam_info.sensorInfo << std::endl;
  std::cout << "Resolution - " << m_cam_info.sensorResolution << std::endl;
  std::cout << "Firmware version - " << m_cam_info.firmwareVersion << std::endl;
  std::cout << "Firmware build time - " << m_cam_info.firmwareBuildTime
            << std::endl
            << std::endl;
}

void FlyCaptureWrapper::readCalibrationFile(const char* calibration_file) {
  readCalibrationParameters(calibration_file);
}

void FlyCaptureWrapper::configureCamera() {
  // Get the current camera configuration
  FlyCapture2::FC2Config config;
  printError(m_cam.GetConfiguration(&config));

  // Modify the configuration
  config.numBuffers = 10;

  // Set the camera configuration
  printError(m_cam.SetConfiguration(&config));

  setCameraProperty();
}

void FlyCaptureWrapper::startCapture() {
  m_cam.StartCapture();
}

void FlyCaptureWrapper::stopCapture() {
  m_cam.StopCapture();
}

unsigned char* FlyCaptureWrapper::getNewFrame() {
  FlyCapture2::Image rawImage;
  printError(m_cam.RetrieveBuffer(&rawImage));
  // FlyCapture2::TimeStamp timestamp = rawImage.GetTimeStamp();

  // Create a converted image
  // FlyCapture2::Image convertedImage;

  // Convert the raw image
  printError(rawImage.Convert(FlyCapture2::PIXEL_FORMAT_MONO8, &m_new_frame));

  // Check image data
  // printf("Image.GetReceivedDataSize() = %d\n", convertedImage.GetReceivedDataSize());
  // printf("Image.GetDataSize() = %d\n", convertedImage.GetDataSize());
  m_new_frame.GetDimensions(&m_n_rows, &m_n_cols, &m_stride, NULL, NULL);

  if (m_undistort) {
    cv::Mat cv_im(m_n_rows, m_n_cols, CV_8UC1, m_new_frame.GetData());
    undistortImage(cv_im, undistorted_frame);
    return undistorted_frame.data;  
  } else {
    return m_new_frame.GetData();
  }

}

int FlyCaptureWrapper::rows() {
  return m_n_rows;
}


int FlyCaptureWrapper::cols() {
  return m_n_cols;
}

int FlyCaptureWrapper::stride() {
  return m_stride;
}
