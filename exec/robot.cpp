#include <fcntl.h>
#include <termios.h>
// #include <ncurses.h>
#include <unistd.h>

#include "stdio.h"
#include "string.h"
#include "robot.h"

#define DEBUG_RX 0

// helper functions
void decomposeInt16 (int16_t data, uint8_t* buffer) {
  // uint8_t* buffer = new uint8_t[2];
  buffer[0] = data & 0x00FF;
  buffer[1] = (data & 0xFF00) >> 8;
  // return buffer;
}


Robot::Robot():
initialized(false)
{
}

Robot::~Robot() {
  closeSerial();
}

bool Robot::initSerial() {
  int fd = 0;
  struct termios options;

  fd = open(SERIAL_PORT, O_RDWR | O_NOCTTY | O_NDELAY);
  if (fd == -1){
    return false;
  }

  fcntl(fd, F_SETFL, 0);    // clear all flags on descriptor, enable direct I/O
  tcgetattr(fd, &options);   // read serial port options
  //set baud rate
  //!!!! PAY ATTENTION TO THE BAUDRATE!!!!
  cfsetispeed(&options, B115200);
  cfsetospeed(&options, B115200);

  /*
  options.c_cflag |= (CLOCAL | CREAD);
  options.c_cflag &= ~PARENB;
  options.c_cflag &= ~CSTOPB;
  options.c_cflag &= ~CSIZE;
  options.c_cflag |= CS8;

  options.c_lflag &= ~ICANON;
  options.c_cc[VTIME] = 1;
  */
  
  options.c_cflag &= ~PARENB;
  options.c_cflag &= ~CSTOPB;
  options.c_cflag &= ~CSIZE;
  options.c_cflag |= CS8;
  // no flow control
  options.c_cflag &= ~CRTSCTS;

  //toptions.c_cflag &= ~HUPCL; // disable hang-up-on-close to avoid reset

  options.c_cflag |= CREAD | CLOCAL;  // turn on READ & ignore ctrl lines
  options.c_iflag &= ~(IXON | IXOFF | IXANY); // turn off s/w flow ctrl

  options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); // make raw
  options.c_oflag &= ~OPOST; // make raw

  // see: http://unixwiz.net/techtips/termios-vmin-vtime.html
  options.c_cc[VMIN]  = 0;
  options.c_cc[VTIME] = 0;
  
  tcsetattr(fd, TCSANOW, &options);

  port_fd = fd;

  initialized = true;
  return true;
}

bool Robot::isInitialized() {
  return initialized;
}


bool Robot::closeSerial() {
  if (port_fd > 0){
    close(port_fd);
    port_fd = -1;
    initialized = false;
    return true;
  }
  return false;
}


// void Robot::setTargetCommand(int16_t target_x, int16_t target_y, int16_t target_angle) {
//   uint8_t checksum = 0;
//   uint8_t buffer[9];
//   uint8_t* decomposed = new uint8_t[2];

//   buffer[0] = START_BYTE;
//   buffer[1] = SET_POS_CMD;
//   decomposeInt16(target_x, decomposed);
//   buffer[2] = decomposed[0];
//   buffer[3] = decomposed[1];
//   decomposeInt16(target_y, decomposed);
//   buffer[4] = decomposed[0];
//   buffer[5] = decomposed[1];
//   decomposeInt16(target_angle, decomposed);
//   buffer[6] = decomposed[0];
//   buffer[7] = decomposed[1];
//   delete decomposed;

//   for (int i = 0; i < 8; i++) {
//     checksum += buffer[i];
//   }
//   checksum &= 0x7F;
//   buffer[8] = checksum;

//   printf("sending command： ");
//   for (int i = 0; i < 9; i++) {
//     printf("%02x ", buffer[i]);
//   }
//   printf("\n");

//   write(port_fd, buffer, 9);
//   sleep(1);
//   tcflush(port_fd, TCIOFLUSH); // important! otherwise old idle signals will be buffered
//   is_idle = false;
// }
void Robot::sendSingleCommand(uint8_t cmd) {
  uint8_t buffer[3];
  uint8_t checksum = 0;

  buffer[0] = START_BYTE;
  buffer[1] = cmd;
  for (int i = 0; i < 2; i++) {
    checksum += buffer[i];
  }
  checksum &= 0x7F;
  buffer[2] = checksum;  

  write(port_fd, buffer, 3);
  usleep(50000); 
}


void Robot::setSpeed(int8_t trans_speed, int8_t rot_speed) {
  uint8_t buffer[5];
  uint8_t checksum = 0;

  buffer[0] = START_BYTE;
  buffer[1] = SET_SPEED_CMD;
  buffer[2] = trans_speed;
  buffer[3] = rot_speed;

  for (int i = 0; i < 4; i++) {
    checksum += buffer[i];
  }
  checksum &= 0x7F;
  buffer[4] = checksum;

  // printf("sending command： ");
  // for (int i = 0; i < 5; i++) {
  //   printf("%02x ", buffer[i]);
  // }
  // printf("\n");

  // for (int i = 0; i < 20; i++) {
  //   buffer[4+20] = 0;
  // }

  write(port_fd, buffer, 5);
  usleep(500); 
  // usleep (7 * 100);  
  // tcdrain(port_fd);
  // sleep(1);
  // tcflush(port_fd, TCIOFLUSH); // important! otherwise old idle signals will be buffered
}


void Robot::setOdometryPosition(int16_t x, int16_t y, int16_t angle) {
  uint8_t buffer[9];
  uint8_t checksum = 0;

  buffer[0] = START_BYTE;
  buffer[1] = SET_POS_CMD;
  decomposeInt16 (x, &buffer[2]);
  decomposeInt16 (y, &buffer[4]);
  decomposeInt16 (angle, &buffer[6]);

  for (int i = 0; i < 8; i++) {
    checksum += buffer[i];
  }
  checksum &= 0x7F;
  buffer[8] = checksum;  

  write(port_fd, buffer, 9);
  usleep(50000);
}

void Robot::setPathPoint(int16_t x, int16_t y, int16_t angle) {
  uint8_t buffer[9];
  uint8_t checksum = 0;

  buffer[0] = START_BYTE;
  buffer[1] = SET_PATH_POINT_CMD;
  decomposeInt16 (x, &buffer[2]);
  decomposeInt16 (y, &buffer[4]);
  decomposeInt16 (angle, &buffer[6]);

  for (int i = 0; i < 8; i++) {
    checksum += buffer[i];
  }
  checksum &= 0x7F;
  buffer[8] = checksum;  

  write(port_fd, buffer, 9);
  usleep(50000);
}

void Robot::toggleManualMode() {
  uint8_t buffer[3];
  uint8_t checksum = 0;

  buffer[0] = START_BYTE;
  buffer[1] = SET_MANUAL_CMD;
  for (int i = 0; i < 2; i++) {
    checksum += buffer[i];
  }
  checksum &= 0x7F;
  buffer[2] = checksum;  

  write(port_fd, buffer, 3);
  usleep(50000);
}

void Robot::setFollowPathCmd() {
  uint8_t buffer[3];
  uint8_t checksum = 0;

  buffer[0] = START_BYTE;
  buffer[1] = SET_FOLLOW_PATH_CMD;
  for (int i = 0; i < 2; i++) {
    checksum += buffer[i];
  }
  checksum &= 0x7F;
  buffer[2] = checksum;  

  write(port_fd, buffer, 3);  
  usleep(50000);
}

void Robot::startTracking() {
  uint8_t buffer[3];
  uint8_t checksum = 0;

  buffer[0] = START_BYTE;
  buffer[1] = START_TRACKING_CMD;
  for (int i = 0; i < 2; i++) {
    checksum += buffer[i];
  }
  checksum &= 0x7F;
  buffer[2] = checksum;  

  write(port_fd, buffer, 3);  
  usleep(500);
}

void Robot::integrateGlobalPosition(int16_t x, int16_t y, int16_t angle) {
  uint8_t buffer[9];
  uint8_t checksum = 0;

  buffer[0] = START_BYTE;
  buffer[1] = SET_GLOBAL_POS_CMD;
  decomposeInt16 (x, &buffer[2]);
  decomposeInt16 (y, &buffer[4]);
  decomposeInt16 (angle, &buffer[6]);

  for (int i = 0; i < 8; i++) {
    checksum += buffer[i];
  }
  checksum &= 0x7F;
  buffer[8] = checksum;  

  write(port_fd, buffer, 9);
  usleep(500);
}





#define WAIT_START_BYTE   0
#define WAIT_DATA_LEN     10
#define WAIT_DATA         20
#define WAIT_CHECKSUM     30

bool Robot::readStatusUntilFinish() {
  char tmp[16];
  char incoming_byte[1];
  static uint8_t checksum = 0;
  static uint8_t n_bytes2wait = 0;

  while(incoming_byte[0] != START_BYTE){
    int ret = read(port_fd, incoming_byte, 1);
    if (ret == -1) {
      return false;
    }
    if (ret == 0) {
      usleep(1000);
    }   
  }
  if (DEBUG_RX) {
    printf("got start byte\n");  
  }
  checksum = START_BYTE;
  // if (incoming_byte == START_BYTE) {
  //   checksum = START_BYTE;
  // } else {
  //   return false;
  // }

  while(read(port_fd, incoming_byte, 1) <= 0){
    usleep(1000);
  }
  checksum += incoming_byte[0];      
  n_bytes2wait = incoming_byte[0];
  if (DEBUG_RX) {
    printf("length = %d\n", n_bytes2wait);
  }
  if (n_bytes2wait > 16) {
    printf("buffer is not enough! data packet invalid!\n");
    return false;
  }

  for (int i = 0; i < n_bytes2wait; i++) {
    while(read(port_fd, incoming_byte, 1) <= 0){
      usleep(1000);
    }
    tmp[i] = incoming_byte[0];
    checksum += incoming_byte[0];  
    if (DEBUG_RX) {
      printf("buffer[i] = %d\n", tmp[i]);  
    }
  }

  while(read(port_fd, incoming_byte, 1) <= 0){
    usleep(1000);
  }

  checksum &= 0x7F;
  if (DEBUG_RX) {
    printf("checksum = %02x\n", checksum);  
  }
  if (incoming_byte[0] == checksum) {
    if (DEBUG_RX) {
      printf("Received new data packet!\n");      
    }
    memcpy(data_buffer, tmp, 16);
    return true;
  } else {
    printf("checksum is wrong!\n");
    return false;
  }

}


bool Robot::readStatus() {
  static uint8_t state = WAIT_START_BYTE;
  static uint8_t checksum = 0;
  static uint8_t byte_index = 0;
  static uint8_t n_bytes2wait = 0;

  uint8_t incoming_byte = 0xFF;
  if (initialized) {
    int ret = read(port_fd, &incoming_byte, 1);
    if (ret != 0) {
      printf("incoming_byte = %02x\n", incoming_byte);
    }
  }

  switch (state) {
    case WAIT_START_BYTE:
      if (incoming_byte == START_BYTE) {
        // printf("got start byte\n");
        state = WAIT_DATA_LEN;
        checksum = START_BYTE;
      }
      break;

    case WAIT_DATA_LEN:
      // printf("getting data length\n");
      checksum += incoming_byte;      
      n_bytes2wait = incoming_byte;
      byte_index = 0;
      state = WAIT_DATA;
      break;

    case WAIT_DATA:
      // printf("getting data\n");
      data_buffer[byte_index++] = incoming_byte;
      checksum += incoming_byte;
      if (byte_index == n_bytes2wait) {
        state = WAIT_CHECKSUM;
      }
      break;

    case WAIT_CHECKSUM:
      // printf("getting checksum\n");
      checksum &= 0x7F; // masked by 127
      // printf("checksum: %02x, incoming: %02x\n", checksum, incoming_byte);
      state = WAIT_START_BYTE;

      if (incoming_byte == checksum) {
        printf("Received new data packet!\n");      
        return true;
      } else {
        printf("checksum is wrong!\n");
      }
      break;
    default:
      break;
  }
  return false;  
}


int16_t Robot::getStatusDataInt16(uint8_t index) {
  int16_t data = 0;
  data += data_buffer[index];
  data += data_buffer[index+1] << 8;
  return data;
}

uint8_t Robot::getStatusDataUint8(uint8_t index) {
  return (uint8_t)(data_buffer[index]);
}

void Robot::assembleStatus() {
  status.x = getStatusDataInt16(0);
  status.y = getStatusDataInt16(2);
  status.angle = getStatusDataInt16(4);
  status.manual_mode = getStatusDataUint8(6);
  status.path_following = getStatusDataUint8(7);
}

RobotStatus& Robot::getAssembledStatus() {
  return status;
}
// void Robot::setParams(int16_t wheel_scale1, int16_t wheel_scale2) {
//   uint8_t checksum = 0;
//   uint8_t buffer[7];
//   uint8_t* decomposed = new uint8_t[2];

//   buffer[0] = START_BYTE;
//   buffer[1] = SET_PARAMS_CMD;
//   decomposeInt16(wheel_scale1, decomposed);
//   buffer[2] = decomposed[0];
//   buffer[3] = decomposed[1];
//   decomposeInt16(wheel_scale2, decomposed);
//   buffer[4] = decomposed[0];
//   buffer[5] = decomposed[1];
//   delete decomposed;

//   for (int i = 0; i < 6; i++) {
//     checksum += buffer[i];
//   }
//   checksum &= 0x7F;
//   buffer[6] = checksum;

//   printf("sending command： ");
//   for (int i = 0; i < 7; i++) {
//     printf("%02x ", buffer[i]);
//   }
//   printf("\n");

//   write(port_fd, buffer, 7);
//   sleep(1);
//   tcflush(port_fd, TCIOFLUSH); // important! otherwise old idle signals will be buffered
// }

// bool Robot::isIdle() {
//   return is_idle == true;
// }

// bool Robot::feedbackThread() {
//   static int n_idle_flags = 0;
//   while (1) {
//     if (port_fd != -1 && is_idle == false) {
//       uint8_t c;
//       if (read(port_fd, &c, 1)) {
//         if (c == IDLE_FLAG) {
//           printf("idle signal received\n");
//           n_idle_flags++;
//           if (n_idle_flags == 10) {
//             is_idle = true;
//             n_idle_flags = 0;
//           }
//         }
//       } else {
//         boost::this_thread::yield();
//       }
//     } else {
//       boost::this_thread::yield();
//     }
//     boost::this_thread::interruption_point();
//   }
// }

// void Robot::waitIdle() {
//   long long prev_time = getCurrentTime();
//   while (true){
//     if (getCurrentTime() - prev_time > 5000){
//       prev_time = getCurrentTime();
//       if (isIdle()) {
//         break;
//       }
//     }
//   }
// }


// void Robot::robotGoto (int16_t target_x, int16_t target_y, int16_t target_angle) {
//   setTargetCommand(target_x, target_y, target_angle*10);
//   waitIdle();
// }

// #ifdef DEBUG
// void waitIdle(Robot* robot) {
//   long long prev_time = getCurrentTime();
//   while (true){
//     if (getCurrentTime() - prev_time > 5000){
//       prev_time = getCurrentTime();
//       if (robot->isIdle()) {
//         break;
//       }
//     }
//   }
// }

// int main(int argc, char const *argv[]) {
//   Robot* omni_robot = new Robot();
//   if (omni_robot->initRobot()) {
//     omni_robot->setRobotSpeed(30, 50);
//     // omni_robot->setParams(157, 307);
//     omni_robot->setTargetCommand(atoi(*(argv+1)), atoi(*(argv+2)), atoi(*(argv+3)));
//     waitIdle(omni_robot);
//     omni_robot->setTargetCommand(0, 300, 0);
//   }
//   sleep(3);
//   return 0;
// }

// #endif
