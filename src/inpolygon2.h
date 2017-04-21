#ifndef _INPOLYGON_H_
#define _INPOLYGON_H_

#include <stdbool.h>

void inpolygon(int nP, float* px, float* py, int nC, float* cx, float* cy, 
              bool* points_in_on, bool* points_in, bool* points_on);

#endif