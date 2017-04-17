#ifndef _INPOLYGON_H_
#define _INPOLYGON_H_

#include <stdbool.h>

void inpolygon(int nP, double* px, double* py, double* cx, double* cy, 
              bool* points_in_on, bool* points_in, bool* points_on);

#endif