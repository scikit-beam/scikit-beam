/* 
 
 pyspec.ccd.ctrans 
 (c) 2010 Stuart Wilkins <stuwilkins@mac.com>
 
 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 
 $Id$
 
 */

#ifndef __CTRANS_H
#define __CTRANS_H

#define HC_OVER_E 12398.4

#define true -1
#define false 0

#ifndef USE_THREADS
#define NTHREADS 1 
#endif

#ifndef NTHREADS
#defile NTHREADS 2
#endif

typedef double _float;
typedef int _int;

typedef struct {
  int xSize;         // X size in pixels.
  int ySize;         // Y size in pixels.
  _float xCen;
  _float yCen;
  _float xPixSize;   // X Pixel Size (microns)
  _float yPixSize;   // Y Pixel Size (microns)
  _float dist;       // Sample - Detector distance. 
} CCD;

typedef struct {
  CCD *ccd;
  _float *anglesp;
  _float *qOutp;
  int ndelgam;
  _float lambda;
  int mode;
  int imstart;
  int imend;
  _float UBI[3][3];
} imageThreadData;

void *processImageThread(void* ptr);
int calcQTheta(_float* diffAngles, _float theta, _float mu, _float *qTheta, _int n, _float lambda);
int calcQPhiFromQTheta(_float *qTheta, _int n, _float chi, _float phi);
int calcDeltaGamma(_float *delgam, CCD *ccd, _float delCen, _float gamCen);
int matmulti(_float *val, int n, _float mat[][3], int skip);
int calcHKLFromQPhi(_float *qPhi, _int n, _float mat[][3]);

unsigned long c_grid3d(double *dout, unsigned long *nout, double *sterr, double *data, double *grid_start, double *grid_stop, int max_data, int *n_grid, int norm_data);

static PyObject* gridder_3D(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject* ccdToQ(PyObject *self, PyObject *args, PyObject *kwargs);

static char *_ctransDoc = \
"Python functions to perform gridding (binning) of experimental data.\n\n";

static PyMethodDef _ctransMethods[] = {
  {"grid3d", (PyCFunction)gridder_3D, METH_VARARGS | METH_KEYWORDS, 
   "Grid the numpy.array object into a regular grid"},
  {"ccdToQ", (PyCFunction)ccdToQ,  METH_VARARGS | METH_KEYWORDS, 
   "Convert CCD image coordinates into Q values"},
  {NULL, NULL, 0, NULL}     /* Sentinel - marks the end of this structure */
};

#endif

