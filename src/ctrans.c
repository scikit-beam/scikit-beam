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
 *
 * This is ctranc.c routine. process_to_q and process_grid
 * functions in the nsls2/recip.py call  ctranc.c routine for
 * fast data analysis.

 */

#ifdef _OPENMP

#include <omp.h>

#else

#define omp_get_thread_num() 0
#define omp_get_max_threads() 0
#define omp_get_num_threads() 1

#endif


#include <stdlib.h>
#include <math.h>

/* Include python and numpy header files */

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <numpy/arrayobject.h>

#include "ctrans.h"

/* Computation functions */
static PyObject* ccdToQ(PyObject *self, PyObject *args, PyObject *kwargs){
  PyArrayObject *angles = NULL;
  PyObject *_angles = NULL;
  PyArrayObject *ubinv = NULL;
  PyObject *_ubinv = NULL;
  PyArrayObject *qOut = NULL;
  CCD ccd;
  npy_intp dims[2];
  npy_intp nimages;
  int retval;

  int mode;

  double lambda;

  double *anglesp = NULL;
  double *qOutp = NULL;
  double *ubinvp = NULL;
  double *delgam = NULL;

  static char *kwlist[] = { "angles", "mode", "ccd_size", "ccd_pixsize",
			                      "ccd_cen", "dist", "wavelength",
			                      "UBinv", NULL };

  if(!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi(ii)(dd)(dd)ddO", kwlist,
				                          &_angles,
				                          &mode,
				                          &ccd.xSize, &ccd.ySize,
				                          &ccd.xPixSize, &ccd.yPixSize,
				                          &ccd.xCen, &ccd.yCen,
				                          &ccd.dist,
				                          &lambda,
				                          &_ubinv)){

    return NULL;
  }

  ccd.size = ccd.xSize * ccd.ySize;

  angles = (PyArrayObject*)PyArray_FROMANY(_angles, NPY_DOUBLE, 2, 2, NPY_ARRAY_IN_ARRAY);
  if(!angles){
    goto cleanup;
  }

  ubinv = (PyArrayObject*)PyArray_FROMANY(_ubinv, NPY_DOUBLE, 2, 2, NPY_ARRAY_IN_ARRAY);
  if(!ubinv){
    goto cleanup;
  }

  ubinvp = (double *)PyArray_DATA(ubinv);

  nimages = PyArray_DIM(angles, 0);

  dims[0] = nimages * ccd.size;
  dims[1] = 3;

  qOut = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  if(!qOut){
    goto cleanup;
  }

  anglesp = (double *)PyArray_DATA(angles);
  qOutp = (double *)PyArray_DATA(qOut);

  // Now create the arrays for delta-gamma pairs
  delgam = (double*)malloc(nimages * ccd.size * sizeof(double) * 2);
  if(!delgam){
    goto cleanup;
  }

  // Ok now we don't touch Python Object ... Release the GIL
  Py_BEGIN_ALLOW_THREADS

  retval = processImages(delgam, anglesp, qOutp, lambda, mode, (unsigned long)nimages,
                         ubinvp, &ccd);

  // Now we have finished with the magic ... Obtain the GIL
  Py_END_ALLOW_THREADS

  if(retval){
    PyErr_SetString(PyExc_RuntimeError, "Error processing images");
    goto cleanup;
  }

  Py_XDECREF(ubinv);
  Py_XDECREF(angles);
  if(delgam) free(delgam);
  return Py_BuildValue("N", qOut);

 cleanup:
  Py_XDECREF(ubinv);
  Py_XDECREF(angles);
  Py_XDECREF(qOut);
  if(delgam) free(delgam);
  return NULL;
}

int processImages(double *delgam, double *anglesp, double *qOutp, double lambda,
                  int mode, unsigned long nimages, double *ubinvp, CCD *ccd){

  int retval = 0;
  unsigned long i;
  double UBI[3][3];

  // Permute the UB matrix into the orientation
  // for the calculations

  for(i=0;i<3;i++){
    UBI[i][0] = -1.0 * ubinvp[2];
    UBI[i][1] = ubinvp[1];
    UBI[i][2] = ubinvp[0];
    ubinvp+=3;
  }

#pragma omp parallel for shared(anglesp, qOutp, delgam, mode)
  for(i=0;i<nimages;i++){
    // Calculate pointer offsets

    double *_anglesp = anglesp + (i * 6);
    double *_qOutp = qOutp + (i * ccd->size * 3);
    double *_delgam = delgam + (i * ccd->size * 2);

    // For each image process
    calcDeltaGamma(_delgam, ccd, _anglesp[0], _anglesp[5]);
    calcQTheta(_delgam, _anglesp[1], _anglesp[4], _qOutp,
	       ccd->size, lambda);
    if(mode > 1){
      calcQPhiFromQTheta(_qOutp, ccd->size, _anglesp[2], _anglesp[3]);
    }
    if(mode == 4){
      calcHKLFromQPhi(_qOutp, ccd->size, UBI);
    }
  }

  return retval;
}

int calcDeltaGamma(double *delgam, CCD *ccd, double delCen, double gamCen){
  // Calculate Delta Gamma Values for CCD
  int i,j;
  double *delgamp = delgam;
  double xPix, yPix;

  xPix = ccd->xPixSize / ccd->dist;
  yPix = ccd->yPixSize / ccd->dist;

  for(j=0;j<ccd->ySize;j++){
    for(i=0;i<ccd->xSize;i++){
      *(delgamp++) = delCen - atan( ((double)j - ccd->yCen) * yPix);
      *(delgamp++) = gamCen - atan( ((double)i - ccd->xCen) * xPix);
    }
  }

  return true;
}

int calcQTheta(double* diffAngles, double theta, double mu, double *qTheta, int n, double lambda){
  // Calculate Q in the Theta frame
  // angles -> Six cicle detector angles [delta gamma]
  // theta  -> Theta value at this detector setting
  // mu     -> Mu value at this detector setting
  // qTheta -> Q Values
  // n      -> Number of values to convert
  int i;
  double *angles;
  double *qt;
  double kl;
  double del, gam;

  angles = diffAngles;
  qt = qTheta;
  kl = 2 * M_PI / lambda;
  for(i=0;i<n;i++){
    del = *(angles++);
    gam = *(angles++);
    *qt = (-1.0 * sin(gam) * kl) - (sin(mu) * kl);

    qt++;
    *qt = (cos(del - theta) * cos(gam) * kl) - (cos(theta) * cos(mu) * kl);

    qt++;
    *qt = (sin(del - theta) * cos(gam) * kl) + (sin(theta) * cos(mu) * kl);

    qt++;
  }

  return true;
}

int calcQPhiFromQTheta(double *qTheta, int n, double chi, double phi){
  double r[3][3];

  r[0][0] = cos(chi);
  r[0][1] = 0.0;
  r[0][2] = -1.0 * sin(chi);
  r[1][0] = sin(phi) * sin(chi);
  r[1][1] = cos(phi);
  r[1][2] = sin(phi) * cos(chi);
  r[2][0] = cos(phi) * sin(chi);
  r[2][1] = -1.0 * sin(phi);
  r[2][2] = cos(phi) * cos(chi);

  matmulti(qTheta, n, r);

  return true;
}

int calcHKLFromQPhi(double *qPhi, int n, double mat[][3]){
  matmulti(qPhi, n, mat);
  return true;
}

int matmulti(double *val, int n, double mat[][3]){
  double *v;
  double qp[3];
  int i,j,k;

  v = val;

  for(i=0;i<n;i++){
    for(k=0;k<3;k++){
      qp[k] = 0.0;
      for(j=0;j<3;j++){
	qp[k] += mat[k][j] * v[j];
      }
    }
    for(k=0;k<3;k++){
      v[k] = qp[k];
    }
    v += 3;
  }

  return true;
}


static PyObject* gridder_3D(PyObject *self, PyObject *args, PyObject *kwargs){
  PyArrayObject *gridout = NULL, *grid2out = NULL, *Nout = NULL, *stderror = NULL;
  PyArrayObject *gridI = NULL;
  PyObject *_dout = NULL, *_d2out = NULL, *_nout = NULL;
  PyObject *_I;

  npy_intp data_size;
  npy_intp dims[3];

  double grid_start[3];
  double grid_stop[3];
  unsigned long grid_nsteps[3];

  int ignore_nan = 0;

  int retval;

  static char *kwlist[] = { "data", "xrange", "yrange", "zrange", "ignore_nan",
                            "gridout", "grid2out", "nout", NULL };

  if(!PyArg_ParseTupleAndKeywords(args, kwargs, "O(ddd)(ddd)(lll)|dOOO", kwlist,
				  &_I,
				  &grid_start[0], &grid_start[1], &grid_start[2],
				  &grid_stop[0], &grid_stop[1], &grid_stop[2],
				  &grid_nsteps[0], &grid_nsteps[1], &grid_nsteps[2],
          &ignore_nan, &_dout, &_d2out, &_nout)){
    return NULL;
  }

  gridI = (PyArrayObject*)PyArray_FROMANY(_I, NPY_DOUBLE, 0, 0, NPY_ARRAY_IN_ARRAY);
  if(!gridI){
    goto error;
  }

  data_size = PyArray_DIM(gridI, 0);
  if(PyArray_DIM(gridI, 1) != 4){
    PyErr_SetString(PyExc_ValueError, "Dimension 1 of array must be 4");
    goto error;
  }

  dims[0] = grid_nsteps[0];
  dims[1] = grid_nsteps[1];
  dims[2] = grid_nsteps[2];

  if(_dout == NULL){
    gridout = (PyArrayObject*)PyArray_ZEROS(3, dims, NPY_DOUBLE, 0);
  } else {
    gridout = (PyArrayObject*)PyArray_FROMANY(_dout, NPY_DOUBLE, 0, 0, NPY_ARRAY_IN_ARRAY);
  }
  if(!gridout){
    goto error;
  }

  if(_d2out == NULL){
    grid2out = (PyArrayObject*)PyArray_ZEROS(3, dims, NPY_DOUBLE, 0);
  } else {
    grid2out = (PyArrayObject*)PyArray_FROMANY(_d2out, NPY_DOUBLE, 0, 0, NPY_ARRAY_IN_ARRAY);
  }
  if(!grid2out){
    goto error;
  }

  if(_nout == NULL){
    Nout = (PyArrayObject*)PyArray_ZEROS(3, dims, NPY_ULONG, 0);
  } else {
    Nout = (PyArrayObject*)PyArray_FROMANY(_nout, NPY_ULONG, 0, 0, NPY_ARRAY_IN_ARRAY);
  }
  if(!Nout){
    goto error;
  }

  stderror = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_DOUBLE);
  if(!stderror){
    goto error;
  }

  // Ok now we don't touch Python Object ... Release the GIL
  Py_BEGIN_ALLOW_THREADS

  retval = c_grid3d((double*)PyArray_DATA(gridout), (double *)PyArray_DATA(grid2out),
                    (unsigned long*)PyArray_DATA(Nout),
                    (double*)PyArray_DATA(stderror), (double*)PyArray_DATA(gridI),
		                grid_start, grid_stop, (unsigned long)data_size, grid_nsteps,
                    ignore_nan);

  // Ok now get the GIL back
  Py_END_ALLOW_THREADS

  if(retval){
    // We had a runtime error
    PyErr_SetString(PyExc_MemoryError, "Could not allocate memory in c_grid3d");
    goto error;
  }

  Py_XDECREF(gridI);
  return Py_BuildValue("NNNN", gridout, grid2out, Nout, stderror);

error:
  Py_XDECREF(gridI);
  Py_XDECREF(gridout);
  Py_XDECREF(grid2out);
  Py_XDECREF(Nout);
  Py_XDECREF(stderror);
  return NULL;
}

int c_grid3d(double *dout, double *d2out, unsigned long *nout, double *stderror, double *data,
             double *grid_start, double *grid_stop, unsigned long max_data,
             unsigned long *n_grid, int ignore_nan){

  unsigned long i, j;
  int n;
  int retval = 0;
  unsigned long grid_size = 0;
  double grid_len[3];

  // Some useful quantities

  grid_size = n_grid[0] * n_grid[1] * n_grid[2];
  for(i=0;i<3; i++){
    grid_len[i] = grid_stop[i] - grid_start[i];
  }

  int max_threads = omp_get_max_threads();
  int num_threads;

  gridderThreadData *threadData = malloc(sizeof(gridderThreadData) * max_threads);
  if(!threadData){
    return 1;
  }

  for(n=0;n<max_threads;n++){
    threadData[n].nout = NULL;
    threadData[n].dout = NULL;
    threadData[n].d2out = NULL;
  }

#pragma omp parallel shared(data, num_threads, threadData, grid_start, grid_len)
  {
    int thread_num = omp_get_thread_num();
    num_threads = omp_get_num_threads();

    double *_d2out;
    double *_dout;
    unsigned long *_nout;

    _d2out = (double*)malloc(sizeof(double) * grid_size);
    _dout = (double *)malloc(sizeof(double) * grid_size);
    _nout = (unsigned long *)malloc(sizeof(unsigned long) * grid_size);

    if((_d2out != NULL) && (_dout != NULL) && (_nout != NULL)){

      // Clear the arrays ....
      for(j=0;j<grid_size;j++){
        _dout[j] = 0.0;
        _d2out[j] = 0.0;
        _nout[j] = 0;
      }

#pragma omp for
      for(i=0;i<max_data;i++){

        double pos_double[3];
        unsigned long grid_pos[3];
        double *data_ptr = data + (i * 4);

        // Check if we have a NaN

        if((ignore_nan == 1) || !isnan(data_ptr[3])){

          // Calculate the relative position in the grid.

          pos_double[0] = (data_ptr[0] - grid_start[0]) / grid_len[0];
          pos_double[1] = (data_ptr[1] - grid_start[1]) / grid_len[1];
          pos_double[2] = (data_ptr[2] - grid_start[2]) / grid_len[2];

          if((pos_double[0] >= 0) && (pos_double[0] < 1) &&
            (pos_double[1] >= 0) && (pos_double[1] < 1) &&
            (pos_double[2] >= 0) && (pos_double[2] < 1)){

            // Calculate the position in the grid
            grid_pos[0] = (int)(pos_double[0] * n_grid[0]);
            grid_pos[1] = (int)(pos_double[1] * n_grid[1]);
            grid_pos[2] = (int)(pos_double[2] * n_grid[2]);

            unsigned long pos =  grid_pos[0] * (n_grid[1] * n_grid[2]);
            pos += grid_pos[1] * n_grid[2];
            pos += grid_pos[2];

            // Store the answer
            _dout[pos] += data_ptr[3];
            _d2out[pos] += (data_ptr[3] * data_ptr[3]);
            _nout[pos]++;
          }
        }
      }

      threadData[thread_num].dout = _dout;
      threadData[thread_num].d2out = _d2out;
      threadData[thread_num].nout = _nout;
    } else {
      retval = 1;
    }

  } // pragma parallel

  if(retval){
    goto error;
  }

  // Now gather the results

  for(n=1;n<num_threads;n++){
    for(j=0;j<grid_size;j++){
      threadData[0].nout[j] += threadData[n].nout[j];
      threadData[0].dout[j] += threadData[n].dout[j];
      threadData[0].d2out[j] += threadData[n].d2out[j];
    }
  }

  // Now copy the outputs to the arrays

  for(j=0;j<grid_size;j++){
    dout[j] += threadData[0].dout[j];
    d2out[j] += threadData[0].d2out[j];
    nout[j] += threadData[0].nout[j];
  }

  // Calculate the stderror

  for(j=0;j<grid_size;j++){
    if(threadData[0].nout[j] == 0){
      stderror[j] = 0.0;
    } else {
      double var = (d2out[j] - pow(dout[j], 2) / nout[j]) / nout[j];
      stderror[j] = pow(var, 0.5) / pow(nout[j], 0.5);
    }
  }

  // Now free the memory.

error:

  for(n=0;n<max_threads;n++){
    if(threadData[n].d2out) free(threadData[n].d2out);
    if(threadData[n].dout) free(threadData[n].dout);
    if(threadData[n].nout) free(threadData[n].nout);
  }

  free(threadData);
  return retval;
}


static PyMethodDef ctrans_methods[] = {
    {"grid3d", (PyCFunction)gridder_3D, METH_VARARGS | METH_KEYWORDS,
     "Grid the numpy.array object into a regular grid"},
    {"ccdToQ", (PyCFunction)ccdToQ,  METH_VARARGS | METH_KEYWORDS,
     "Convert CCD image coordinates into Q values"},
    {NULL, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "ctrans",
    "Python functions to perform gridding (binning) of experimental data.\n\n",
    -1, // we keep state in global vars
    ctrans_methods,
};

PyObject* PyInit_ctrans(void) {

  PyObject *module = PyModule_Create(&moduledef);
  if(!module){
    return NULL;
  }

  import_array();

  return module;
}

#else // We have Python 2 ...

PyMODINIT_FUNC initctrans(void){
  PyObject *module = Py_InitModule3("ctrans", ctrans_methods, _ctransDoc);
  if(!module){
    return;
  }

  import_array();
}
#endif
