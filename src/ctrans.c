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
 *
 *
 * This is ctranc.c routine. process_to_q and process_grid
 * functions in the nsls2/recip.py call  ctranc.c routine for
 * fast data analysis.
 
 */

#include <stdlib.h>
#include <math.h>

/* Include python and numpy header files */

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <numpy/arrayobject.h>

/* If useing threading then import pthreads */
#ifdef USE_THREADS
#include <pthread.h>
#include <unistd.h>
#endif

#include "ctrans.h"

/* Set global variable to indicate number of threads to create */
unsigned int _n_threads = 1;

/* Computation functions */
static PyObject* ccdToQ(PyObject *self, PyObject *args, PyObject *kwargs){
  static char *kwlist[] = { "angles", "mode", "ccd_size", "ccd_pixsize",
			                      "ccd_cen", "dist", "wavelength",
			                      "UBinv", "n_threads", NULL };
  PyArrayObject *angles = NULL;
  PyObject *_angles = NULL;
  PyArrayObject *ubinv = NULL;
  PyObject *_ubinv = NULL;
  PyArrayObject *qOut = NULL;
  CCD ccd;
  npy_intp dims[2];
  npy_intp nimages;

  int mode;

  double lambda;

  double *anglesp = NULL;
  double *qOutp = NULL;
  double *ubinvp = NULL;
  double *delgam = NULL;

  unsigned int n_threads = _n_threads;

  if(!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi(ii)(dd)(dd)ddO|i", kwlist,
				  &_angles,
				  &mode,
				  &ccd.xSize, &ccd.ySize,
				  &ccd.xPixSize, &ccd.yPixSize,
				  &ccd.xCen, &ccd.yCen,
				  &ccd.dist,
				  &lambda,
				  &_ubinv,
          &n_threads)){
    return NULL;
  }

#ifdef USE_THREADS

  if(n_threads > MAX_THREADS){
    PyErr_SetString(PyExc_ValueError, "n_threads > MAX_THREADS");
    goto cleanup;
  }

  if(n_threads < 1){
    n_threads = _n_threads;
  }

#else

  if(n_threads > 1){
    PyErr_SetString(PyExc_RuntimeError, "Multithreading support is not compiled in");
    goto cleanup;
  }

#endif

  ccd.size = ccd.xSize * ccd.ySize;

  angles = (PyArrayObject*)PyArray_FROMANY(_angles, NPY_DOUBLE, 2, 2, NPY_ARRAY_IN_ARRAY);
  if(!angles){
    PyErr_SetString(PyExc_ValueError, "angles must be a 2-D array of floats");
    goto cleanup;
  }
  
  ubinv = (PyArrayObject*)PyArray_FROMANY(_ubinv, NPY_DOUBLE, 2, 2, NPY_ARRAY_IN_ARRAY);
  if(!ubinv){
    PyErr_SetString(PyExc_ValueError, "ubinv must be a 2-D array of floats");
    goto cleanup;
  }

  ubinvp = (double *)PyArray_DATA(ubinv);
  
  nimages = PyArray_DIM(angles, 0);

  dims[0] = nimages * ccd.size;
  dims[1] = 3;

  qOut = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  if(!qOut){
    PyErr_SetString(PyExc_MemoryError, "Could not allocate memory (qOut)");
    goto cleanup;
  }

  
  anglesp = (double *)PyArray_DATA(angles);
  qOutp = (double *)PyArray_DATA(qOut);

  // Now create the arrays for delta-gamma pairs
  delgam = (double*)malloc(nimages * ccd.size * sizeof(double) * 2);
  if(!delgam){
    PyErr_SetString(PyExc_MemoryError, "Could not allocate memory (delgam)");
    goto cleanup;
  }


  // Ok now we don't touch Python Object ... Release the GIL
  Py_BEGIN_ALLOW_THREADS

  if(processImages(delgam, anglesp, qOutp, lambda, mode, (unsigned long)nimages, 
                n_threads, ubinvp, &ccd)){
    PyErr_SetString(PyExc_RuntimeError, "Processing data failed");
    goto cleanup;
  }

  // Now we have finished with the magic ... Obtain the GIL
  Py_END_ALLOW_THREADS

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
                  int mode, unsigned long nimages, unsigned int n_threads, double *ubinvp,
                  CCD *ccd){

  int retval = 0;
  unsigned long i, j, t;
  double *_delgam = delgam;
  unsigned long stride = nimages / n_threads;
  imageThreadData threadData[MAX_THREADS];
  double UBI[3][3];
#ifdef USE_THREADS
  pthread_t thread[MAX_THREADS];
#endif

  for(i=0;i<3;i++){
    UBI[i][0] = -1.0 * ubinvp[2];
    UBI[i][1] = ubinvp[1];
    UBI[i][2] = ubinvp[0];
    ubinvp+=3;
  }

  for(t=0;t<n_threads;t++){
    // Setup threads
    // Allocate memory for delta/gamma pairs
    
    threadData[t].ccd = ccd;
    threadData[t].anglesp = anglesp;
    threadData[t].qOutp = qOutp;
    threadData[t].lambda = lambda;
    threadData[t].mode = mode;
    threadData[t].imstart = stride * t;
    threadData[t].delgam = _delgam;
    threadData[t].retval = 0;

    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	       threadData[t].UBI[j][i] = UBI[j][i];
      }
    }

    if(t == (n_threads - 1)){
      threadData[t].imend = nimages;
    } else {
      threadData[t].imend = stride * (t + 1);
    }

#ifdef USE_THREADS

    // Start the thread processing 
    
    if(pthread_create(&thread[t], NULL,
		            	     processImageThread,
			                 (void*) &threadData[t])){
      return -1;
    }

#else

    processImageThread((void *) &threadData[t]);

#endif

    anglesp += (6 * stride);
    _delgam += (ccd->size * 2 * stride);
    qOutp += (ccd->size * 3 * stride);
  }

#ifdef USE_THREADS

  for(t=0;t<n_threads;t++){
    if(pthread_join(thread[t], NULL)){
      return -1;
    }
     
    // Check the thread retval

    if(threadData[t].retval){
      retval = -1;
    }
  }

#endif
  return retval;
}

void *processImageThread(void* ptr){
  imageThreadData *data;
  unsigned long i;
  data = (imageThreadData*) ptr;

  for(i=data->imstart;i<data->imend;i++){
    // For each image process
    calcDeltaGamma(data->delgam, data->ccd, data->anglesp[0], data->anglesp[5]);
    calcQTheta(data->delgam, data->anglesp[1], data->anglesp[4], data->qOutp,
	       data->ccd->size, data->lambda);
    if(data->mode > 1){
      calcQPhiFromQTheta(data->qOutp, data->ccd->size,
			 data->anglesp[2], data->anglesp[3]);
    }
    if(data->mode == 4){
      calcHKLFromQPhi(data->qOutp, data->ccd->size, data->UBI);
    }
    data->anglesp += 6;
    data->qOutp += (data->ccd->size * 3);
    data->delgam += (data->ccd->size * 2);
  }
  
  // Set the retval to zero to show sucsessful processing
  data->retval = 0;

#ifdef USE_THREADS
  pthread_exit(NULL);
#else
  return NULL;
#endif
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
  PyArrayObject *gridout = NULL, *Nout = NULL, *stderror = NULL;
  PyArrayObject *gridI = NULL, *meanout = NULL;
  PyObject *_I;
  
  npy_intp data_size;
  npy_intp dims[3];
  
  double grid_start[3];
  double grid_stop[3];
  unsigned long grid_nsteps[3];
  unsigned long n_outside;

  unsigned int n_threads = _n_threads;
  int retval;

  static char *kwlist[] = { "data", "xrange", "yrange", "zrange", "n_threads", NULL }; 

  if(!PyArg_ParseTupleAndKeywords(args, kwargs, "O(ddd)(ddd)(lll)|i", kwlist, 
				  &_I,
				  &grid_start[0], &grid_start[1], &grid_start[2],
				  &grid_stop[0], &grid_stop[1], &grid_stop[2],
				  &grid_nsteps[0], &grid_nsteps[1], &grid_nsteps[2],
          &n_threads)){
    return NULL;
  }

#ifdef USE_THREADS

  if(n_threads > MAX_THREADS){
    PyErr_SetString(PyExc_ValueError, "n_threads > MAX_THREADS");
    goto error;
  }

  if(n_threads < 1){
    n_threads = _n_threads;
  }

#else

  if(n_threads > 1){
    PyErr_SetString(PyExc_RuntimeError, "Multithreading support is not compiled in");
    goto error;
  }

#endif
  
  gridI = (PyArrayObject*)PyArray_FROMANY(_I, NPY_DOUBLE, 0, 0, NPY_ARRAY_IN_ARRAY);
  if(!gridI){
    PyErr_SetString(PyExc_MemoryError, "Could not allocate memory (gridI)");
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

  gridout = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_DOUBLE);
  if(!gridout){
    PyErr_SetString(PyExc_MemoryError, "Could not allocate memory (gridout)");
    goto error;
  }

  Nout = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_ULONG);
  if(!Nout){
    PyErr_SetString(PyExc_MemoryError, "Could not allocate memory (Nout)");
    goto error;
  }

  stderror = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_DOUBLE);
  if(!stderror){
    PyErr_SetString(PyExc_MemoryError, "Could not allocate memory (stderror)");
    goto error;
  }

  meanout = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_DOUBLE);
  if(!meanout){
    PyErr_SetString(PyExc_MemoryError, "Could not allocate memory (meanout)");
    goto error;
  }
 
  // Ok now we don't touch Python Object ... Release the GIL
  Py_BEGIN_ALLOW_THREADS

  retval = c_grid3d((double*)PyArray_DATA(gridout), (unsigned long*)PyArray_DATA(Nout),
                    (double*)PyArray_DATA(meanout), (double*)PyArray_DATA(stderror), 
                    (double*)PyArray_DATA(gridI), &n_outside,
		                grid_start, grid_stop, (unsigned long)data_size, grid_nsteps, 1,
                    n_threads);

  // Ok now get the GIL back
  Py_END_ALLOW_THREADS

  if(retval){
    // We had a runtime error
    PyErr_SetString(PyExc_RuntimeError, "Gridding process failed to run");
    goto error;
  }

  Py_XDECREF(gridI);
  return Py_BuildValue("NNNNl", gridout, meanout, Nout, stderror, n_outside);

error:
  Py_XDECREF(gridI);
  Py_XDECREF(gridout);
  Py_XDECREF(meanout);
  Py_XDECREF(Nout);
  Py_XDECREF(stderror);
  return NULL;
}

int c_grid3d(double *dout, unsigned long *nout, double *mout,
             double *stderror, double *data, unsigned long *n_outside,
             double *grid_start, double *grid_stop, unsigned long max_data,
             unsigned long *n_grid, int norm, unsigned int n_threads){

  unsigned long i, j;
  unsigned long grid_size = 0;
  double grid_len[3];
  unsigned long stride;
	
  // Some useful quantities

  grid_size = n_grid[0] * n_grid[1] * n_grid[2];
  for(i = 0;i < 3; i++){
    grid_len[i] = grid_stop[i] - grid_start[i];
  }

  // If we do this with threads .. we can do map reduce

#ifdef USE_THREADS
  pthread_t thread[MAX_THREADS];
#endif

  gridderThreadData threadData[MAX_THREADS];

  // Allocate arrays for standard error calculation
 
  for(i=0;i<n_threads;i++){
    threadData[i].Mk = NULL;
    threadData[i].Qk = NULL;
    threadData[i].dout = NULL;
    threadData[i].nout = NULL;
  }

  stride = max_data / n_threads;
  for(i=0;i<n_threads;i++){
    threadData[i].Qk = (double*)malloc(sizeof(double) * grid_size);
    if(!threadData[i].Qk){
      goto error;
    }
    if(i > 0){
      threadData[i].Mk = (double*)malloc(sizeof(double) * grid_size);
      if(!threadData[i].Mk){
        goto error;
      }
      threadData[i].dout = (double *)malloc(sizeof(double) * grid_size);
      if(!threadData[i].dout){
        goto error;
      }
      threadData[i].nout = (unsigned long *)malloc(sizeof(unsigned long) * grid_size);
      if(!threadData[i].nout){
        goto error;
      }
    } else {
      threadData[i].dout = dout;
      threadData[i].nout = nout;
      threadData[i].Mk = mout;
    }

    // Clear the arrays ....
    for(j=0;j<grid_size;j++){
      threadData[i].Mk[j] = 0.0;
      threadData[i].Qk[j] = 0.0;
      threadData[i].dout[j] = 0.0;
      threadData[i].nout[j] = 0;
    }
    threadData[i].n_outside = 0;

    // Setup entry points for threads
    threadData[i].start = stride * i;
    if(i == (n_threads - 1)){
      threadData[i].end = max_data;
    } else {
      threadData[i].end = stride * (i + 1);
    }

    threadData[i].data = data;
    threadData[i].n_grid = n_grid;
    threadData[i].grid_start = grid_start;
    threadData[i].grid_len = grid_len;
    threadData[i].retval = 0;

#ifdef USE_THREADS
    pthread_create(&thread[i], NULL,
                   grid3DThread,
                   (void*) &threadData[i]);
#else
    grid3DThread((void *) &threadData[i]);
#endif
  }

#ifdef USE_THREADS
  // Wait for threads to finish and then join them
  for(i=0;i<n_threads;i++){
    if(pthread_join(thread[i], NULL)){
      goto error;
    }
    if(threadData[i].retval){
      goto error;
    }
  }
#endif

  // Combine results
  if(n_threads > 1){
    for(j=0;j<grid_size;j++){
      threadData[0].Qk[j] = (threadData[0].Qk[j] * threadData[0].nout[j]);
      threadData[0].Mk[j] = (threadData[0].Mk[j] * threadData[0].nout[j]);
    }
      //fprintf(stderr, "0 : Qk = %f, N = %ld\n", threadData[0].Qk[j], threadData[0].nout[j]);
  }

  for(i=1;i<n_threads;i++){
    for(j=0;j<grid_size;j++){
      threadData[0].nout[j] += threadData[i].nout[j];
      threadData[0].dout[j] += threadData[i].dout[j];
      threadData[0].Qk[j] += (threadData[i].Qk[j] * threadData[i].nout[j]);
      threadData[0].Mk[j] += (threadData[i].Mk[j] * threadData[i].nout[j]);
      //fprintf(stderr, "%ld : Qk = %f, N = %ld\n", i,threadData[i].Qk[j], threadData[i].nout[j]);
    }
    threadData[0].n_outside += threadData[i].n_outside;
  }

  // Calculate the sterror

  for(j=0;j<grid_size;j++){
    if(threadData[0].nout[j] == 0){
      threadData[0].Mk[j] = 0.0;
    } else {
      if(n_threads > 1){
        threadData[0].Mk[j] = threadData[0].Mk[j] / threadData[0].nout[j];
        threadData[0].Qk[j] = threadData[0].Qk[j] / threadData[0].nout[j];
      }
      if(threadData[0].nout[j] > 1){
        stderror[j] = pow(threadData[0].Qk[j] / 
            (threadData[0].nout[j] - 1), 0.5) / pow(threadData[0].nout[j], 0.5);
      } else {
        stderror[j] = 0.0;
      }
      if(norm){
        threadData[0].Mk[j] = threadData[0].dout[j] / threadData[0].nout[j];
      }
    }
  }

  // Store the number of elements outside the grid
  
  *n_outside = threadData[0].n_outside;

  // Now free the memory.

  for(i=0;i<n_threads;i++){
    free(threadData[i].Qk);
    if(i > 0){
      free(threadData[i].Mk);
      free(threadData[i].dout);
      free(threadData[i].nout);
    }
  }
  return 0;

error:
  for(i=0;i<n_threads;i++){
    if(threadData[i].Qk) free(threadData[i].Qk); 
    if(i > 0){
      if(threadData[i].dout) free(threadData[i].dout); 
      if(threadData[i].nout) free(threadData[i].nout); 
      if(threadData[i].Mk) free(threadData[i].Mk);
    }
  }
  return -1;
}

void* grid3DThread(void *ptr){
  gridderThreadData* data = (gridderThreadData*)ptr;
  double pos_double[3];
  unsigned long grid_pos[3];
  double *grid_start = data->grid_start; 
  double *grid_len = data->grid_len; 
  unsigned long *n_grid = data->n_grid;
  double *Mk = data->Mk;
  double *Qk = data->Qk;
  double *data_ptr = data->data;
  double *dout = data->dout;
  unsigned long *nout = data->nout;
  unsigned long pos = 0;
  
  unsigned long i;

  data_ptr = data_ptr + (data->start * 4);
  for(i=data->start; i<data->end; i++){
    // Calculate the relative position in the grid.
    pos_double[0] = (*data_ptr - grid_start[0]) / grid_len[0];
    data_ptr++;
    pos_double[1] = (*data_ptr - grid_start[1]) / grid_len[1];
    data_ptr++;
    pos_double[2] = (*data_ptr - grid_start[2]) / grid_len[2];
    if((pos_double[0] >= 0) && (pos_double[0] < 1) &&
       (pos_double[1] >= 0) && (pos_double[1] < 1) &&
       (pos_double[2] >= 0) && (pos_double[2] < 1)){
      
      data_ptr++;

      // Calculate the position in the grid
      grid_pos[0] = (int)(pos_double[0] * n_grid[0]);
      grid_pos[1] = (int)(pos_double[1] * n_grid[1]);
      grid_pos[2] = (int)(pos_double[2] * n_grid[2]);
      
      pos =  grid_pos[0] * (n_grid[1] * n_grid[2]);
      pos += grid_pos[1] * n_grid[2];
      pos += grid_pos[2];


      // Store the answer
      dout[pos] = dout[pos] + *data_ptr;
      nout[pos] = nout[pos] + 1;

      // Calculate the standard deviation quantities

      Qk[pos] = Qk[pos] + ((nout[pos] - 1) * pow(*data_ptr - Mk[pos],2) / nout[pos]);
      Mk[pos] = Mk[pos] + ((*data_ptr - Mk[pos]) / nout[pos]);
      //fprintf(stderr, "Qk = %f, Mk = %f\n", Qk[pos], Mk[pos]);

      // Increment pointer
      data_ptr++;
    } else {
      data->n_outside++;
      data_ptr+=2;
    }
  }

  return NULL;
}

long nproc(void) {
  long _n;
#ifdef USE_THREADS 

  _n = sysconf(_SC_NPROCESSORS_ONLN);
  if(_n > MAX_THREADS){
    _n = MAX_THREADS;
  }

#else

  _n = 1;

#endif

  return _n;
}

static PyObject* get_threads(PyObject *self, PyObject *args){
  return PyLong_FromLong((long)_n_threads);
}

static PyObject* set_threads(PyObject *self, PyObject *args){

#ifdef USE_THREADS

  int threads;

  if(!PyArg_ParseTuple(args, "i", &threads)){
    return NULL;
  }
  if(threads > MAX_THREADS){
    PyErr_SetString(PyExc_ValueError, "Requested number of threads > MAX_THREADS");
    return NULL;
  }
  _n_threads = threads;
  Py_RETURN_NONE;

#else

  PyErr_SetString(PyExc_RuntimeError, "Module has been compiled not to use threads.");
  return NULL;

#endif
}

static PyMethodDef ctrans_methods[] = {
    {"grid3d", (PyCFunction)gridder_3D, METH_VARARGS | METH_KEYWORDS,
     "Grid the numpy.array object into a regular grid"},
    {"ccdToQ", (PyCFunction)ccdToQ,  METH_VARARGS | METH_KEYWORDS,
     "Convert CCD image coordinates into Q values"},
    {"get_threads", (PyCFunction)get_threads, METH_VARARGS,
      "Return the number of threads used"},
    {"set_threads", (PyCFunction)set_threads, METH_VARARGS,
      "Set the number of threads used"},
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
  _n_threads = nproc();

  return module;
}

#else // We have Python 2 ... 

PyMODINIT_FUNC initctrans(void){
  PyObject *module = Py_InitModule3("ctrans", ctrans_methods, _ctransDoc);
  if(!module){
    return;
  }

  import_array();
  _n_threads = nproc();
}
#endif
