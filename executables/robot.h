#ifndef OMNI_ROBOT_H
#define OMNI_ROBOT_H

#include "stdint.h"


#ifdef ON_MAC
#define SERIAL_PORT "/dev/tty.HC-06-DevB"
// #define SERIAL_PORT "/dev/tty.usbmodem901661"
#endif

#ifdef ON_TEGRA
#define SERIAL_PORT "/dev/ttyACM0"
#endif

struct RobotStatus {
  RobotStatus():
  x(-9999),
  y(-9999),
  angle(-9999),
  manual_mode(0),
  path_following(0),
  timestamp(-1)
  {
  }

  RobotStatus& operator= (RobotStatus& arg) {
    x = arg.x;
    y = arg.y;
    angle = arg.angle;
    manual_mode = arg.manual_mode;
    path_following = arg.path_following;
    timestamp = arg.timestamp;
  }
  int16_t x;
  int16_t y;
  int16_t angle;
  uint8_t manual_mode;
  uint8_t path_following;
  long long timestamp;
};


#define START_BYTE 0x41

#define SET_SPEED_CMD          0xCA
#define SET_POS_CMD            0xCB
#define SET_PARAMS_CMD         0xCD
#define SET_MANUAL_CMD         0xC1
#define SET_PATH_POINT_CMD     0xC2
#define SET_FOLLOW_PATH_CMD    0xC3
#define CLEAR_PATH_CMD         0xC4
#define START_TRACKING_CMD     0xC5
#define SET_GLOBAL_POS_CMD     0xC6

class Robot {
public:
  Robot();
  ~Robot();
  bool initSerial();
  bool closeSerial();
  bool isInitialized();
  bool readStatusUntilFinish();
  bool readStatus();
  int16_t getStatusDataInt16(uint8_t index);
  uint8_t getStatusDataUint8(uint8_t index);
  void assembleStatus();
  RobotStatus& getAssembledStatus();
  
  // bool initRobot();
  // void setTargetCommand(int16_t target_x, int16_t target_y, int16_t target_angle);
  // void waitIdle();
  // void robotGoto(int16_t target_x, int16_t target_y, int16_t target_angle);
  void sendSingleCommand(uint8_t cmd);
  void setSpeed(int8_t rot_speed, int8_t trans_speed);
  void setPathPoint(int16_t x, int16_t y, int16_t angle);
  void setOdometryPosition(int16_t x, int16_t y, int16_t angle);
  void setFollowPathCmd();
  void toggleManualMode();  
  void startTracking();
  void integrateGlobalPosition(int16_t x, int16_t y, int16_t angle);

  // void setParams(int16_t wheel_scale1, int16_t wheel_scale2);
  // bool isIdle();

private:
  int port_fd;
  bool initialized;
  uint8_t data_buffer[16];
  RobotStatus status;
};

#endif
