#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include "stdio.h"

using namespace cv;


#define SQUARE_SIZE_X 121.5f / 10.0f
#define SQUARE_SIZE_Y 91.0f / 8.0f

#define CHESS_ORIGIN_X_IN_ROBOT -50.25f
#define CHESS_ORIGIN_Y_IN_ROBOT 322.f

int main(int argc, char const *argv[]) {
  Size boardSize;
  boardSize.width = 9;
  boardSize.height = 7;

  Mat view = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

  // imshow("image", view);
  // waitKey(0);
  vector<Point2f> pointbuf;
  bool found = findChessboardCorners(view, boardSize, pointbuf,
                                    CV_CALIB_CB_ADAPTIVE_THRESH |
                                    CV_CALIB_CB_FAST_CHECK);
                                    // CV_CALIB_CB_NORMALIZE_IMAGE); //don't normalize


  if (found) {
    cornerSubPix(view, pointbuf, Size(11, 11), Size(-1, -1),
            TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
    drawChessboardCorners(view, boardSize, Mat(pointbuf), found);

    for (int i = 0; i < pointbuf.size(); i++) {
      cv::putText(view, std::to_string(i), cv::Point(pointbuf[i].x, pointbuf[i].y), cv::FONT_HERSHEY_SIMPLEX, 1.00f, cv::Scalar(255, 0, 0), 1);
    }

    imshow("corners", view);
    waitKey(0);
  } else {
    printf("didn't find anything!\n");
  }

  printf("size pointbuf = %ld\n", pointbuf.size());
  Point2f top_left = pointbuf[pointbuf.size()-1];
  Point2f top_right = pointbuf[pointbuf.size()-boardSize.width];

  std::cout << "top_left = " << top_left << std::endl;
  std::cout << "top_right = " << top_right << std::endl;

  double length = sqrt((top_right.x - top_left.x) * (top_right.x - top_left.x) + 
                       (top_right.y - top_left.y) * (top_right.y - top_left.y));

  double mm_per_pixel = ((boardSize.width-1) * SQUARE_SIZE_X) / length;
  printf("%f mm/pixel\n", mm_per_pixel);

  double x_axis_x = (top_right.x - top_left.x) / length; 
  double x_axis_y = (top_right.y - top_left.y) / length; 
  double y_axis_x = -x_axis_y;
  double y_axis_y = x_axis_x;
  double origin_x = top_left.x * mm_per_pixel;
  double origin_y = top_left.y * mm_per_pixel;

  Mat T_camera_chess = Mat::eye(3, 3, CV_64FC1);
  T_camera_chess.at<double>(0, 0) = x_axis_x;
  T_camera_chess.at<double>(0, 1) = y_axis_x;
  T_camera_chess.at<double>(0, 2) = origin_x;
  T_camera_chess.at<double>(1, 0) = x_axis_y;
  T_camera_chess.at<double>(1, 1) = y_axis_y;
  T_camera_chess.at<double>(1, 2) = origin_y;

  std::cout << "T_camera_chess = \n" << T_camera_chess << std::endl;

  // TODO: modify this
  // double chess_origin_x_in_robot = 0.0f;
  // double chess_origin_y_in_robot = 500.0f;


  Mat T_robot_chess = Mat::eye(3, 3, CV_64FC1);
  T_robot_chess.at<double>(0, 0) = 1;
  T_robot_chess.at<double>(0, 1) = 0;
  T_robot_chess.at<double>(0, 2) = CHESS_ORIGIN_X_IN_ROBOT;
  T_robot_chess.at<double>(1, 0) = 0;
  T_robot_chess.at<double>(1, 1) = -1;
  T_robot_chess.at<double>(1, 2) = CHESS_ORIGIN_Y_IN_ROBOT;

  // T_world_robot = T_world_camera * T_camera_chess * T_chess_robot;
  Mat T_camera_robot = T_camera_chess * T_robot_chess.inv();
  // Mat T_robot_camera = T_camera_robot.inv();

  std::cout << "T_camera_robot = \n" << T_camera_robot << std::endl;

  FILE* fp = fopen("robot_camera_calibration.txt", "w");
  fprintf(fp, "%f %f %f\n", T_camera_robot.at<double>(0, 0), T_camera_robot.at<double>(0, 1), T_camera_robot.at<double>(0, 2));
  fprintf(fp, "%f %f %f\n", T_camera_robot.at<double>(1, 0), T_camera_robot.at<double>(1, 1), T_camera_robot.at<double>(1, 2));
  fprintf(fp, "%f\n", mm_per_pixel);
  fclose(fp);

  // for (int i = 0; i < boardSize.height ; i++) {
  //   for (int j = 0; j < boardSize.width; j++) {
  //     if (i+1) < 

  //   }
  // }





}