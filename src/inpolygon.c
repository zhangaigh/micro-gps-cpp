/*-----------------------------------
  * in polygon
  * MEX file
  * Fast detection points inside a polygonal region
  *
  * [IN,ON,IN_strict] = InPolygon(px,py,cx,cy);
  *
  * input: px - nPM x nPN (Matrix) X-coordinates of the points to be tested
  *        py - nPM x nPN (Matrix) Y-coordinates of the points to be tested
  *        cx - nC  x 1   (Vector) X-coordinates of the polygon
  *        cy - nC  x 1   (Vector) Y-coordinates of the polygon
  * output:
  *        IN           - boolean matrix of 1 if px(i,j),py(i,j) is
  *                       in or on the polygon defined by cx,cy
  *        ON           - boolean matrix of 1 if px(i,j),py(i,j) is
  *                       on the polygon defined by cx,cy
  *        IN_strict    - boolean matrix of 1 if px(i,j),py(i,j) is
  *                       exclusively in  the polygon defined by cx,cy
  *
 -----------------------------------*/

/*************************************************************************
 %  To compile
 *  mex InPolygon.c
 *
 *  for Windows
 *  mex -output InPolygon.dll InPolygon.c

% Example
    X=[0 0.5 1 0.5 0];Y=[0 0 0.5 1 0.5];
    [points_in_on,points_on,points_in] = InPolygon(X,Y,[0 1 1 0],[0 0 1 1])
 *
 To check the efficiency of the routine:
    npoints = 1e5;
    L = linspace(0,2.*pi,6); xv = cos(L)';yv = sin(L)';
    xv = [xv ; xv(1)]; yv = [yv ; yv(1)];
    x = randn(npoints,1); y = randn(npoints,1);
    tic,in = inpolygon(x,y,xv,yv);
    disp(['Time needed by MatLab inpolygon routine = ' num2str(toc)])
    tic,[points_in_on,points_on,points_in] = InPolygon(x,y,xv,yv);
    disp(['Time needed by InPolygon routine = ' num2str(toc)])
    plot(xv,yv,x(in),y(in),'r+',x(~in),y(~in),'bo')
**************************************************************************/

/* Authors:                                                              */
/* A. David Redish      email: adr at nsma dot arizona dot edu           */
/* Guillaume Jacquenot  email: guillaume dot jacquenot dot gmail dot com */

// #include "mex.h"
#include <math.h>
#include "inpolygon.h"
// #include <matrix.h>

#define EPS 1.0e-10

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) > (b) ? (b) : (a))

void inpolygon(int nP, double* px, double* py, double* cx, double* cy, 
              bool* points_in_on, bool* points_in, bool* points_on) {
    
  // int nPM, nPN;
  // int nP, nC;
  int nC;
  int iP, iC;
  int ind;
  // double *px, *py, *cx, *cy;
  // bool *points_in_on, *points_in, *points_on;

  double xmin, xmax, ymin, ymax;
  double ax, bx, ay, by;
  double nIntersect, intersecty, tmp;

  /* unpack inputs */
  // nPM = mxGetM(pINP[0]);
  // nPN = mxGetN(pINP[0]);
  // nP = nPM * nPN;
  // nP = 1;
  // px = (double *)mxGetPr(pINP[0]);
  // py = (double *)mxGetPr(pINP[1]);

  // nC = mxGetM(pINP[2]) * mxGetN(pINP[2]);
  nC = 4;
  // cx = (double *)mxGetPr(pINP[2]);
  // cy = (double *)mxGetPr(pINP[3]);

  ax = cx[0];
  ay = cy[0];
  bx = cx[nC - 1];
  by = cy[nC - 1];

  /* Check if the polygon is closed */
  /* If so, the number of points is decreased */
  if ((ax == bx) && (ay == by)) nC--;

  // /* pack outputs */

  /* calculate */
  /* first find min and max x and y */
  xmin = xmax = cx[0];
  ymin = ymax = cy[0];
  for (iC = 0; iC < nC; iC++) {
    if (cx[iC] < xmin) xmin = cx[iC];
    if (cx[iC] > xmax) xmax = cx[iC];
    if (cy[iC] < ymin) ymin = cy[iC];
    if (cy[iC] > ymax) ymax = cy[iC];
  }
  /* for each point is it in the polygon or not */
  for (iP = 0; iP < nP; iP++) {
    /*mexPrintf("\niP=%d",iP);*/
    points_on[iP] = 0;
    points_in[iP] = 0;
    points_in_on[iP] = 0;
    /* count the number of intersections */
    nIntersect = 0.0;
    /* is it even in the rectangle defined by the polygon? */
    if (px[iP] < xmin) continue;
    if (px[iP] > xmax) continue;
    if (py[iP] < ymin) continue;
    if (py[iP] > ymax) continue;

    for (iC = 0; iC < nC; iC++) {
      /* does the line PQ intersect the line AB? */
      if (iC == (nC - 1)) {
        ax = cx[nC - 1];
        ay = cy[nC - 1];
        bx = cx[0];
        by = cy[0];
      } else {
        ax = cx[iC];
        ay = cy[iC];
        bx = cx[iC + 1];
        by = cy[iC + 1];
      }
      if (ax == bx)
      /* Vertical points*/
      {
        if (px[iP] == ax) {
          /* ensure order correct */
          if (ay > by) {
            tmp = ay;
            ay = by;
            by = tmp;
          }
          if ((py[iP] >= ay) && (py[iP] <= by)) {
            points_on[iP] = 1;
            nIntersect = 0.0;
            break;
          }
        }
      }
      /* Non Vertical points*/
      else {
        if ((px[iP] < MIN(ax, bx)) || (MAX(ax, bx) < px[iP])) {
          continue;
        }
        intersecty = ay + (px[iP] - ax) / (bx - ax) * (by - ay);
        if (fabs(intersecty - py[iP]) < EPS) {
          points_on[iP] = 1;
          nIntersect = 0.0;
          break;
        } else if ((intersecty < py[iP]) &&
                   ((ax == px[iP]) || (bx == px[iP]))) {
          if (ax == px[iP]) {
            if (iC == 0) {
              ind = nC - 1;
            } else {
              ind = iC - 1;
            }
            if ((MIN(bx, cx[ind]) < px[iP]) && (px[iP] < MAX(bx, cx[ind]))) {
              nIntersect++;
            }
          }
        } else if (intersecty < py[iP]) {
          nIntersect++;
        }
      }
    } /* for iC */
    /*Check if the contour polygon is closed*/
    points_in[iP] = (nIntersect - 2 * floor(nIntersect / 2));
    points_in_on[iP] = MAX(points_on[iP], points_in[iP]);
  } /* for iP */
}
