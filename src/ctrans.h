/*
 * Copyright (c) 2014, Brookhaven Science Associates, Brookhaven
 * National Laboratory. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright
 *   notice this list of conditions and the following disclaimer in
 *   the documentation and/or other materials provided with the
 *   distribution.
 *
 * * Neither the name of the Brookhaven Science Associates, Brookhaven
 *   National Laboratory nor the names of its contributors may be used
 *   to endorse or promote products derived from this software without
 *   specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef _CTRANS_H
#define _CTRANS_H

#define HC_OVER_E 12398.4

#define true -1
#define false 0

#ifndef MAX_THREADS
#define MAX_THREADS 128
#endif

typedef struct {
  int xSize;         // X size in pixels.
  int ySize;         // Y size in pixels.
  long int size;          // Total size for convinience
  double xCen;
  double yCen;
  double xPixSize;   // X Pixel Size (microns)
  double yPixSize;   // Y Pixel Size (microns)
  double dist;       // Sample - Detector distance.
} CCD;

typedef struct {
  double *dout;
  double *d2out;
  unsigned long *nout;
} gridderThreadData;

int calcQTheta(double* diffAngles, double theta, double mu,
               double *qTheta, int n, double lambda);
int calcQPhiFromQTheta(double *qTheta, int n, double chi, double phi);
int calcDeltaGamma(double *delgam, CCD *ccd, double delCen, double gamCen);
int matmulti(double *val, int n, double mat[][3]);
int calcHKLFromQPhi(double *qPhi, int n, double mat[][3]);

int processImages(double *delgam, double *anglesp, double *qOutp, double lambda,
                  int mode, unsigned long nimages, double *ubinvp, CCD *ccd);

int c_grid3d(double *dout, double *d2out, unsigned long *nout, double *sterr, double *data,
             double *grid_start, double *grid_stop, unsigned long max_data,
             unsigned long *n_grid, int ignore_nan);

static PyObject* gridder_3D(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject* ccdToQ(PyObject *self, PyObject *args, PyObject *kwargs);

#if PY_MAJOR_VERSION < 3
static char *_ctransDoc = \
"Python functions to perform gridding (binning) of experimental data.\n\n";
#endif

#endif
