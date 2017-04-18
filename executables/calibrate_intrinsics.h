#ifndef _CALIBRATE_INTRINSICS_H
#define _CALIBRATE_INTRINSICS_H

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>


void calibrateImageSequence(std::vector<char*> imageList, const char* outputFilename);

void readCalibrationParameters(const char* calibration_file);

void undistortImage(cv::Mat& image, cv::Mat& undistorted_image);

#endif