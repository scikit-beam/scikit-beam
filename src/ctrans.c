/*
 
 This is ctranc.c routine. process_to_q and process_grid
 functions in the nsls2/recip.py call  ctranc.c routine for
 fast data analysis.
 
 */
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <math.h>
#ifdef USE_THREADS
#include <pthread.h>
#endif

/* Set up */

#define HC_OVER_E 12398.4

#define true -1
#define false 0

#ifndef USE_THREADS
#define NTHREADS 1
#endif

#ifndef NTHREADS
#define NTHREADS 2
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

/* Computation functions */
static PyObject* ccdToQ(PyObject *self, PyObject *args, PyObject *kwargs){
  static char *kwlist[] = { "angles", "mode", "ccd_size", "ccd_pixsize",
			    "ccd_cen", "dist", "wavelength",
			    "UBinv", "outarray", NULL };
  PyObject *angles = NULL;
  PyObject *_angles = NULL;
  PyObject *_ubinv = NULL;
  PyObject *ubinv = NULL;
  PyObject *_outarray = NULL;
  PyObject *qOut = NULL;
  CCD ccd;
  npy_intp dims[2];
  npy_intp nimages;
  int i, j, t, stride;
  int ndelgam;
  int mode;

  _float lambda;

  _float *anglesp;
  _float *qOutp;
  _float *ubinvp;
  _float UBI[3][3];

#ifdef USE_THREADS
  pthread_t thread[NTHREADS];
  int iret[NTHREADS];
#endif
  imageThreadData threadData[NTHREADS];

  if(!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi(ii)(dd)(dd)ddO|O", kwlist,
				  &_angles,
				  &mode,
				  &ccd.xSize, &ccd.ySize,
				  &ccd.xPixSize, &ccd.yPixSize,
				  &ccd.xCen, &ccd.yCen,
				  &ccd.dist,
				  &lambda,
				  &_ubinv,
				  &_outarray)){
    return NULL;
  }

  angles = PyArray_FROMANY(_angles, NPY_DOUBLE, 2, 2, NPY_IN_ARRAY);
  if(!angles){
    PyErr_SetString(PyExc_ValueError, "angles must be a 2-D array of floats");
    goto cleanup;
  }
  
  ubinv = PyArray_FROMANY(_ubinv, NPY_DOUBLE, 2, 2, NPY_IN_ARRAY);
  if(!ubinv){
    PyErr_SetString(PyExc_ValueError, "ubinv must be a 2-D array of floats");
    goto cleanup;
  }

  ubinvp = (_float *)PyArray_DATA(ubinv);
  for(i=0;i<3;i++){
    UBI[i][0] = -1.0 * ubinvp[2];
    UBI[i][1] = ubinvp[1];
    UBI[i][2] = ubinvp[0];
    ubinvp+=3;
  }
  
  nimages = PyArray_DIM(angles, 0);
  ndelgam = ccd.xSize * ccd.ySize;

  dims[0] = nimages * ndelgam;
  dims[1] = 4;
  if(!_outarray){
    // Create new numpy array
    // fprintf(stderr, "**** Creating new array\n");
    qOut = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if(!qOut){
      goto cleanup;
    }
  } else {
    qOut = PyArray_FROMANY(_outarray, NPY_DOUBLE, 2, 2, NPY_INOUT_ARRAY);
    if(!qOut){
      PyErr_SetString(PyExc_ValueError, "outarray must be a 2-D array of floats");
      goto cleanup;
    }
    if(PyArray_Size(qOut) != (4 * nimages * ndelgam)){
      PyErr_SetString(PyExc_ValueError, "outarray is of the wrong size");
      goto cleanup;
    }
  }
  anglesp = (_float *)PyArray_DATA(angles);
  qOutp = (_float *)PyArray_DATA(qOut);

  stride = nimages / NTHREADS;
  for(t=0;t<NTHREADS;t++){
    // Setup threads
    // Allocate memory for delta/gamma pairs
    
    threadData[t].ccd = &ccd;
    threadData[t].anglesp = anglesp;
    threadData[t].qOutp = qOutp;
    threadData[t].ndelgam = ndelgam;
    threadData[t].lambda = lambda;
    threadData[t].mode = mode;
    threadData[t].imstart = stride * t;
    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	       threadData[t].UBI[j][i] = UBI[j][i];
      }
    }
    if(t == (NTHREADS - 1)){
      threadData[t].imend = nimages;
    } else {
      threadData[t].imend = stride * (t + 1);
    }

#ifdef USE_THREADS
    iret[t] = pthread_create( &thread[t], NULL,
			      processImageThread,
			      (void*) &threadData[t]);
#else
    processImageThread((void *) &threadData[t]);
#endif
    anglesp += (6 * stride);
    qOutp += (ndelgam * 4 * stride);
  }

#ifdef USE_THREADS
  for(t=0;t<NTHREADS;t++){
    if(pthread_join(thread[t], NULL)){
      fprintf(stderr, "ERROR : Cannot join thread %d", t);
    }
  }
#endif

  Py_XDECREF(ubinv);
  Py_XDECREF(angles);
  return Py_BuildValue("N", qOut);

 cleanup:
  Py_XDECREF(ubinv);
  Py_XDECREF(angles);
  Py_XDECREF(qOut);
  return NULL;
}

void *processImageThread(void* ptr){
  imageThreadData *data;
  int i;
  _float *delgam;
  data = (imageThreadData*) ptr;
  delgam = (_float*)malloc(data->ndelgam * sizeof(_float) * 2);
  if(!delgam){
    fprintf(stderr, "MALLOC ERROR\n");
#ifdef USE_THREADS
    pthread_exit(NULL);
#endif
  }
  
  for(i=data->imstart;i<data->imend;i++){
    // For each image process
    calcDeltaGamma(delgam, data->ccd, data->anglesp[0], data->anglesp[5]);
    calcQTheta(delgam, data->anglesp[1], data->anglesp[4], data->qOutp,
	       data->ndelgam, data->lambda);
    if(data->mode > 1){
      calcQPhiFromQTheta(data->qOutp, data->ndelgam,
			 data->anglesp[2], data->anglesp[3]);
    }
    if(data->mode == 4){
      calcHKLFromQPhi(data->qOutp, data->ndelgam, data->UBI);
    }
    data->anglesp+=6;
    data->qOutp+=(data->ndelgam * 4);
  }
  free(delgam);
#ifdef USE_THREADS
  pthread_exit(NULL);
#endif
}

int calcQTheta(_float* diffAngles, _float theta, _float mu, _float *qTheta, _int n, _float lambda){
  // Calculate Q in the Theta frame
  // angles -> Six cicle detector angles [delta gamma]
  // theta  -> Theta value at this detector setting
  // mu     -> Mu value at this detector setting
  // qTheta -> Q Values
  // n      -> Number of values to convert
  _int i;
  _float *angles;
  _float *qt;
  _float kl;
  _float del, gam;

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
    qt++;
  }
  
  return true;
}

int calcQPhiFromQTheta(_float *qTheta, _int n, _float chi, _float phi){
  _float r[3][3];

  r[0][0] = cos(chi);
  r[0][1] = 0.0;
  r[0][2] = -1.0 * sin(chi);
  r[1][0] = sin(phi) * sin(chi);
  r[1][1] = cos(phi);
  r[1][2] = sin(phi) * cos(chi);
  r[2][0] = cos(phi) * sin(chi);
  r[2][1] = -1.0 * sin(phi);
  r[2][2] = cos(phi) * cos(chi);

  matmulti(qTheta, n, r, 1);
  
  return true;
}

int calcHKLFromQPhi(_float *qPhi, _int n, _float mat[][3]){
  matmulti(qPhi, n, mat, 1);
  return true;
}

int matmulti(_float *val, int n, _float mat[][3], int skip){
  _float *v;
  _float qp[3];
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
    v += skip;
  }

  return true;
}

int calcDeltaGamma(_float *delgam, CCD *ccd, _float delCen, _float gamCen){
  // Calculate Delta Gamma Values for CCD
  int i,j;
  _float *delgamp;
  _float xPix, yPix;

  xPix = ccd->xPixSize / ccd->dist;
  yPix = ccd->yPixSize / ccd->dist;
  delgamp = delgam;

  for(j=0;j<ccd->ySize;j++){
    for(i=0;i<ccd->xSize;i++){
      *(delgamp++) = delCen - atan( ((_float)j - ccd->yCen) * yPix);
      *(delgamp++) = gamCen - atan( ((_float)i - ccd->xCen) * xPix);
    }
  }

  return true;
}

static PyObject* gridder_3D(PyObject *self, PyObject *args, PyObject *kwargs){
  PyObject *gridout = NULL, *Nout = NULL, *standarderror = NULL;
  PyObject *gridI = NULL;
  PyObject *_I;
  
  static char *kwlist[] = { "data", "xrange", "yrange", "zrange", "norm", NULL };
  
  npy_intp data_size;
  npy_intp dims[3];
  
  double grid_start[3];
  double grid_stop[3];
  int grid_nsteps[3];
  int norm_data = 0;
  
  unsigned long n_outside;
  
  if(!PyArg_ParseTupleAndKeywords(args, kwargs, "O(ddd)(ddd)(iii)|i", kwlist,
				  &_I,
				  &grid_start[0], &grid_start[1], &grid_start[2],
				  &grid_stop[0], &grid_stop[1], &grid_stop[2],
				  &grid_nsteps[0], &grid_nsteps[1], &grid_nsteps[2],
				  &norm_data)){
    return NULL;
  }
  
  gridI = PyArray_FROMANY(_I, NPY_DOUBLE, 0, 0, NPY_IN_ARRAY);
  if(!gridI){
    goto cleanup;
  }
  
  data_size = PyArray_DIM(gridI, 0);
  
  dims[0] = grid_nsteps[0];
  dims[1] = grid_nsteps[1];
  dims[2] = grid_nsteps[2];

  gridout = PyArray_ZEROS(3, dims, NPY_DOUBLE, 0);
  if(!gridout){
    goto cleanup;
  }
  Nout = PyArray_ZEROS(3, dims, NPY_ULONG, 0);
  if(!Nout){
    goto cleanup;
  }
  standarderror = PyArray_ZEROS(3, dims, NPY_DOUBLE, 0);
  if(!standarderror){
    goto cleanup;
  }
  
  n_outside = c_grid3d(PyArray_DATA(gridout), PyArray_DATA(Nout),
		       PyArray_DATA(standarderror), PyArray_DATA(gridI),
		       grid_start, grid_stop, data_size, grid_nsteps, norm_data);
  
  Py_XDECREF(gridI);
  return Py_BuildValue("NNNl", gridout, Nout, standarderror, n_outside);
  
 cleanup:
  Py_XDECREF(gridI);
  Py_XDECREF(gridout);
  Py_XDECREF(Nout);
  Py_XDECREF(standarderror);
  return NULL;
}

unsigned long c_grid3d(double *dout, unsigned long *nout, double *standarderror, double *data,
		       double *grid_start, double *grid_stop, int max_data,
		       int *n_grid, int norm_data){
  int i;
  unsigned long *nout_ptr;
  double *dout_ptr;
  double *data_ptr;
  
  double *Mk = NULL;
  double *Qk = NULL;
  int grid_size = 0;

  double pos_double[3];
  double grid_len[3], grid_step[3];
  int grid_pos[3];
  int pos = 0;
  unsigned long n_outside = 0;
	
  // Some useful quantities

  grid_size = n_grid[0] * n_grid[1] * n_grid[2];

  // Allocate arrays for standard error calculation
  
  if(standarderror){
    Mk = (double*)malloc(sizeof(double) * grid_size);
    if(!Mk){
      return n_outside;
    }
    Qk = (double*)malloc(sizeof(double) * grid_size);
    if(!Mk){
      return n_outside;
    }
  }

  dout_ptr = dout;
  nout_ptr = nout;
	
  data_ptr = data;
  for(i = 0;i < 3; i++){
    grid_len[i] = grid_stop[i] - grid_start[i];
    grid_step[i] = grid_len[i] / (n_grid[i]);
  }
	
  for(i = 0; i < max_data ; i++){
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

      if(standarderror){
      	if(nout[pos] == 1){
      	  Mk[pos] = *data_ptr;
      	  Qk[pos] = 0.0;
      	} else {
      	  Qk[pos] = Qk[pos] + ((nout[pos] - 1) * pow(*data_ptr - Mk[pos],2) / nout[pos]);
      	  Mk[pos] = Mk[pos] + ((*data_ptr - Mk[pos]) / nout[pos]);
      	}
      }

      // Increment pointer
      data_ptr++;
    } else {
      n_outside++;
      data_ptr+=2;
    }
  }
  
  // Calculate mean by dividing by the number of data points in each
  // voxel

  if(norm_data){
    for(i = 0; i < grid_size; i++){
      if(nout[i] > 0){
	       dout[i] = dout[i] / nout[i];
      } else {
	       dout[i] = 0.0;
      }
    }
  }

  // Calculate the sterror
  
  if(standarderror){
    for(i=0;i<grid_size;i++){
      if(nout[i] > 1){
      	// standard deviation of the sample distribution
      	standarderror[i] = pow(Qk[pos] / (nout[i] - 1), 0.5) / pow(nout[i], 0.5);
      }
    }
  }

  if(Mk){
    free(Mk);
  }
  if(Qk){
    free(Qk);
  }
	
  return n_outside;
}

// // This is the python 2 version
// PyMODINIT_FUNC initctrans(void)  {
// 	(void) Py_InitModule3("ctrans", _ctransMethods, _ctransDoc);
// 	import_array();  // Must be present for NumPy.  Called first after above line.
// }
struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

static PyObject *
error_out(PyObject *m) {
    struct module_state *st = GETSTATE(m);
    PyErr_SetString(st->error, "something bad happened");
    return NULL;
}

static PyMethodDef _ctransMethods[] = {
  {NULL, NULL, 0, NULL}     /* Sentinel - marks the end of this structure */
};

static PyMethodDef ctrans_methods[] = {
    {"error_out", (PyCFunction)error_out, METH_NOARGS, NULL},
    {"grid3d", (PyCFunction)gridder_3D, METH_VARARGS | METH_KEYWORDS,
     "Grid the numpy.array object into a regular grid"},
    {"ccdToQ", (PyCFunction)ccdToQ,  METH_VARARGS | METH_KEYWORDS,
     "Convert CCD image coordinates into Q values"},
    {NULL, NULL}
};

#if PY_MAJOR_VERSION >= 3

static int ctrans_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int ctrans_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "ctrans",
        NULL,
        sizeof(struct module_state),
        ctrans_methods,
        NULL,
        ctrans_traverse,
        ctrans_clear,
        NULL
};

#define INITERROR return NULL

PyObject *
PyInit_ctrans(void)

#else
#define INITERROR return

void
initctrans(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("ctrans", ctrans_methods);
#endif
    import_array();

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("ctrans.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
