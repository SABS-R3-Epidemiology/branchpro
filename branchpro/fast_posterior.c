#include <Python.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <float.h>


#include <limits.h>






// Digamma function code copied from from online
///////////////////////////////////////////////////////
// http://www.mymathlib.com/c_source/functions/gamma_beta/digamma_function.c
////////////////

//                         Internally Defined Routines                        //

double DiGamma_Function(double x);
long double xDiGamma_Function(long double x);

static long double xDiGamma(long double x);
static long double xDiGamma_Asymptotic_Expansion(long double x);


//                         Internally Defined Constants                       //

static double cutoff = 171.0;
static long double const pi = 3.14159265358979323846264338L;
static long double const g =  9.6565781537733158945718737389L;
static long double const a[] = { +1.144005294538510956673085217e+4L,
                                 -3.239880201523183350535979104e+4L,
                                 +3.505145235055716665660834611e+4L,
                                 -1.816413095412607026106469185e+4L,
                                 +4.632329905366668184091382704e+3L,
                                 -5.369767777033567805557478696e+2L,
                                 +2.287544733951810076451548089e+1L,
                                 -2.179257487388651155600822204e-1L,
                                 +1.083148362725893688606893534e-4L
                              };

////////////////////////////////////////////////////////////////////////////////
// double DiGamma_Function( double x )                                        //
//                                                                            //
//  Description:                                                              //
//     This function uses the derivative of the log of Lanczos's expression   //
//     for the Gamma function to calculate the DiGamma function for x > 1 and //
//     x <= cutoff = 171.  An asymptotic expression for the DiGamma function  //
//     for x > cutoff.  The reflection formula                                //
//                      DiGamma(x) = DiGamma(1+x) - 1/x                       //
//     for 0 < x < 1. and the reflection formula                              //
//                DiGamma(x) = DiGamma(1-x) - pi * cot(pi*x)                  //
//     for x < 0.                                                             //
//     The DiGamma function has singularities at the nonpositive integers.    //
//     At a singularity, DBL_MAX is returned.                                 //
//                                                                            //
//  Arguments:                                                                //
//     double x   Argument of the DiGamma function.                           //
//                                                                            //
//  Return Values:                                                            //
//     If x is a nonpositive integer then DBL_MAX is returned otherwise       //
//     DiGamma(x) is returned.                                                //
//                                                                            //
//  Example:                                                                  //
//     double x, psi;                                                         //
//                                                                            //
//     psi = DiGamma_Function( x );                                           //
////////////////////////////////////////////////////////////////////////////////
double DiGamma_Function(double x)
{
   long double psi = xDiGamma_Function((long double) x);

   if (fabsl(psi) < DBL_MAX) return (double) psi;
   return (psi < 0.0L) ? -DBL_MAX : DBL_MAX;
}


////////////////////////////////////////////////////////////////////////////////
// long double xDiGamma_Function( long double x )                             //
//                                                                            //
//  Description:                                                              //
//     This function uses the derivative of the log of Lanczos's expression   //
//     for the Gamma function to calculate the DiGamma function for x > 1 and //
//     x <= cutoff = 171.  An asymptotic expression for the DiGamma function  //
//     for x > cutoff.  The reflection formula                                //
//                      DiGamma(x) = DiGamma(1+x) - 1/x                       //
//     for 0 < x < 1. and the reflection formula                              //
//                DiGamma(x) = DiGamma(1-x) - pi * cot(pi*x)                  //
//     for x < 0.                                                             //
//     The DiGamma function has singularities at the nonpositive integers.    //
//     At a singularity, LDBL_MAX is returned.                                //
//                                                                            //
//  Arguments:                                                                //
//     long double x   Argument of the DiGamma function.                      //
//                                                                            //
//  Return Values:                                                            //
//     If x is a nonpositive integer then LDBL_MAX is returned otherwise      //
//     DiGamma(x) is returned.                                                //
//                                                                            //
//  Example:                                                                  //
//     long double x, psi;                                                    //
//                                                                            //
//     psi = xDiGamma_Function( x );                                          //
////////////////////////////////////////////////////////////////////////////////
long double xDiGamma_Function(long double x)
{
   long double sin_x, cos_x;
   long int ix;

             // For a positive argument (x > 0)                 //
             //    if x <= cutoff return Lanzcos approximation  //
             //    otherwise return Asymptotic approximation.   //

   if ( x > 0.0L )
      if (x <= cutoff)
         if ( x >= 1.0L) return xDiGamma(x);
         else return xDiGamma( x + 1.0L ) - (1.0L / x);
      else return xDiGamma_Asymptotic_Expansion(x);

                  // For a nonpositive argument (x <= 0). //
               // If x is a singularity then return LDBL_MAX. //

   if ( x > -(long double)LONG_MAX) {
      ix = (long int) x;
      if ( x == (long double) ix) return LDBL_MAX;
   }

   sin_x = sinl(pi * x);
   if (sin_x == 0.0L) return LDBL_MAX;
   cos_x = cosl(pi * x);
   if (fabsl(cos_x) == 1.0L) return LDBL_MAX;

            // If x is not a singularity then return DiGamma(x). //

   return xDiGamma(1.0L - x) - pi * cos_x / sin_x;
}


////////////////////////////////////////////////////////////////////////////////
// static long double xDiGamma( long double x )                               //
//                                                                            //
//  Description:                                                              //
//     This function uses the derivative of the log of Lanczos's expression   //
//     for the Gamma function to calculate the DiGamma function for x > 1 and //
//     x <= cutoff = 171.                                                     //
//                                                                            //
//  Arguments:                                                                //
//     double x   Argument of the Gamma function.                             //
//                                                                            //
//  Return Values:                                                            //
//     DiGamma(x)                                                             //
//                                                                            //
//  Example:                                                                  //
//     long double x;                                                         //
//     long double psi;                                                       //
//                                                                            //
//     g = xDiGamma( x );                                                     //
////////////////////////////////////////////////////////////////////////////////
static long double xDiGamma(long double x) {

   long double lnarg = (g + x - 0.5L);
   long double temp;
   long double term;
   long double numerator = 0.0L;
   long double denominator = 0.0L;
   int const n = sizeof(a) / sizeof(long double);
   int i;

   for (i = n-1; i >= 0; i--) {
      temp = x + (long double) i;
      term = a[i] / temp;
      denominator += term;
      numerator += term / temp;
   }
   denominator += 1.0L;

   return logl(lnarg) - (g / lnarg) - numerator / denominator;
}


////////////////////////////////////////////////////////////////////////////////
// static long double xDiGamma_Asymptotic_Expansion( long double x )          //
//                                                                            //
//  Description:                                                              //
//     This function estimates DiGamma(x) by evaluating the asymptotic        //
//     expression:                                                            //
//         DiGamma(x) ~ ln(x) - (1/2) x +                                     //
//                        Sum B[2j] / [ 2j * x^(2j) ], summed over            //
//     j from 1 to 3 and where B[2j] is the 2j-th Bernoulli number.           //
//                                                                            //
//  Arguments:                                                                //
//     long double x   Argument of the DiGamma function. The argument x must  //
//                     be positive.                                           //
//                                                                            //
//  Return Values:                                                            //
//     DiGamma(x) where x > cutoff.                                           //
//                                                                            //
//  Example:                                                                  //
//     long double x;                                                         //
//     long double g;                                                         //
//                                                                            //
//     g = xDiGamma_Asymptotic_Expansion( x );                                //
////////////////////////////////////////////////////////////////////////////////

// Bernoulli numbers B(2j) / 2j: B(2)/2,B(4)/4,B(6)/6,...,B(20)/20.  Only     //
//  B(2)/2,..., B(6)/6 are currently used.                                    //

static const long double B[] = {   1.0L / (long double)(6 * 2 ),
                                  -1.0L / (long double)(30 * 4 ),
                                   1.0L / (long double)(42 * 6 ),
                                  -1.0L / (long double)(30 * 8 ),
                                   5.0L / (long double)(66 * 10 ),
                                -691.0L / (long double)(2730 * 12 ),
                                   7.0L / (long double)(6 * 14 ),
                               -3617.0L / (long double)(510 * 16 ),
                               43867.0L / (long double)(796 * 18 ),
                             -174611.0L / (long double)(330 * 20 )
                           };

static const int n = sizeof(B) / sizeof(long double);

static long double xDiGamma_Asymptotic_Expansion(long double x ) {
   const int  m = 3;
   long double term[3];
   long double sum = 0.0L;
   long double xx = x * x;
   long double xj = x;
   long double digamma = logl(xj) - 1.0L / (xj + xj);
   int i;

   xj = xx;
   for (i = 0; i < m; i++) { term[i] = B[i] / xj; xj *= xx; }
   for (i = m - 1; i >= 0; i--) sum += term[i];
   return digamma - sum;
}
/////////////////////////////////////////////


// posterior functions
/////////////


static PyObject *
update_neg_bin_ll_onestep(PyObject *self, PyObject *args) {

  double phi;
  double Rt;
  int window_size;
  PyObject *py_slice_cases;
  PyObject *py_tau_window;
  PyObject *py_sum_tau_window;
  PyObject *py_log_tau_window;
  PyObject *py_ll_normalizing;

  if (!PyArg_ParseTuple(args, "ddiOOOOO", &phi, &Rt, &window_size,
      &py_slice_cases, &py_tau_window, &py_sum_tau_window, &py_log_tau_window,
      &py_ll_normalizing))
    return NULL;

  double slice_cases[window_size];
  int i = 0;
  PyObject *iterator = PyObject_GetIter(py_slice_cases);
  PyObject *item;
  if (iterator == NULL) {
    return NULL;
  }
  while ((item = PyIter_Next(iterator))) {
    slice_cases[i] = PyFloat_AsDouble(item);
    Py_DECREF(item);
    i += 1;
  }
  Py_DECREF(iterator);


  double tau_window[window_size];
  i = 0;
  iterator = PyObject_GetIter(py_tau_window);
  if (iterator == NULL) {
    return NULL;
  }
  while ((item = PyIter_Next(iterator))) {
    tau_window[i] = PyFloat_AsDouble(item);
    Py_DECREF(item);
    i += 1;
  }
  Py_DECREF(iterator);

  double sum_tau_window[window_size];
  i = 0;
  iterator = PyObject_GetIter(py_sum_tau_window);
  if (iterator == NULL) {
    return NULL;
  }
  while ((item = PyIter_Next(iterator))) {
    sum_tau_window[i] = PyFloat_AsDouble(item);
    Py_DECREF(item);
    i += 1;
  }
  Py_DECREF(iterator);

  double log_tau_window[window_size];
  i = 0;
  iterator = PyObject_GetIter(py_log_tau_window);
  if (iterator == NULL) {
    return NULL;
  }
  while ((item = PyIter_Next(iterator))) {
    log_tau_window[i] = PyFloat_AsDouble(item);
    Py_DECREF(item);
    i += 1;
  }
  Py_DECREF(iterator);

  double ll_normalizing[window_size];
  i = 0;
  iterator = PyObject_GetIter(py_ll_normalizing);
  if (iterator == NULL) {
    return NULL;
  }
  while ((item = PyIter_Next(iterator))) {
    ll_normalizing[i] = PyFloat_AsDouble(item);
    Py_DECREF(item);
    i += 1;
  }
  Py_DECREF(iterator);


  // for (int index=0; index<window_size; index++) {
  //   printf("arr[%d] = %f\n", index, slice_cases[index]);
  // }

  double ll_diff = 0;

  double loggamma_slice_cases_phi[window_size];
  for (int index=0; index<window_size; index++) {
    loggamma_slice_cases_phi[index] = lgamma(slice_cases[index] + 1.0/phi) - lgamma(1.0/phi);
  }

  for (int index=0; index<window_size; index++) {
    ll_diff += (loggamma_slice_cases_phi[index] - log(phi) / phi);
  }

  for (int index=0; index<window_size; index++) {
    ll_diff += log(Rt) * slice_cases[index];
  }

  double log_phi_r_tau_window[window_size];
  for (int index=0; index<window_size; index++) {
    log_phi_r_tau_window[index] = log(1.0/phi + Rt * sum_tau_window[index]);
  }

  for (int index=0; index<window_size; index++) {
    ll_diff += slice_cases[index] * log_tau_window[index];
  }

  for (int index=0; index<window_size; index++) {
    ll_diff -= (slice_cases[index] + 1.0/phi) * log_phi_r_tau_window[index];
  }

  for (int index=0; index<window_size; index++) {
    ll_diff -= ll_normalizing[index];
  }

  return PyFloat_FromDouble(ll_diff);
}


static PyObject *
update_neg_bin_deriv_onestep(PyObject *self, PyObject *args) {

  double phi;
  double Rt;
  int window_size;
  PyObject *py_tau_window;
  PyObject *py_sum_tau_window;
  PyObject *py_slice_cases;

  if (!PyArg_ParseTuple(args, "ddiOOO", &phi, &Rt, &window_size,
      &py_tau_window, &py_sum_tau_window, &py_slice_cases))
    return NULL;

  double slice_cases[window_size];
  int i = 0;
  PyObject *iterator = PyObject_GetIter(py_slice_cases);
  PyObject *item;
  if (iterator == NULL) {
    return NULL;
  }
  while ((item = PyIter_Next(iterator))) {
    slice_cases[i] = PyFloat_AsDouble(item);
    Py_DECREF(item);
    i += 1;
  }
  Py_DECREF(iterator);


  double tau_window[window_size];
  i = 0;
  iterator = PyObject_GetIter(py_tau_window);
  if (iterator == NULL) {
    return NULL;
  }
  while ((item = PyIter_Next(iterator))) {
    tau_window[i] = PyFloat_AsDouble(item);
    Py_DECREF(item);
    i += 1;
  }
  Py_DECREF(iterator);


  double sum_tau_window[window_size];
  i = 0;
  iterator = PyObject_GetIter(py_sum_tau_window);
  if (iterator == NULL) {
    return NULL;
  }
  while ((item = PyIter_Next(iterator))) {
    sum_tau_window[i] = PyFloat_AsDouble(item);
    Py_DECREF(item);
    i += 1;
  }
  Py_DECREF(iterator);


  double dLl_i = 0;
  double dLl_phi_step = 0;

  double inv_phi_r_tau_window[window_size];
  double inv_phi_r_tau_window2[window_size];
  double log_phi_r_tau_window[window_size];

  for (int index=0; index<window_size; index++) {
    inv_phi_r_tau_window[index] = sum_tau_window[index] * 1.0 / (1.0/ phi + Rt * sum_tau_window[index]);
    inv_phi_r_tau_window2[index] = 1.0 / (1.0/ phi + Rt * sum_tau_window[index]);
    log_phi_r_tau_window[index] = log(1.0/ phi + Rt * sum_tau_window[index]);
  }

  for (int index=0; index<window_size; index++) {
    dLl_i += 1.0 / Rt * slice_cases[index];
    dLl_i -= (slice_cases[index] + 1.0/phi) * inv_phi_r_tau_window[index];
  }

  for (int index=0; index<window_size; index++) {
    dLl_phi_step += DiGamma_Function(slice_cases[index] + 1.0/phi) - DiGamma_Function(1.0/phi) + log(1.0 / phi) + 1.0;
    dLl_phi_step -= log_phi_r_tau_window[index];
    dLl_phi_step -= (slice_cases[index] + 1.0 / phi) * inv_phi_r_tau_window2[index];
  }


  // Return as a python list of floats
  return Py_BuildValue("[ff]", dLl_i, dLl_phi_step);
}




// Methods table for python
static PyMethodDef PosteriorMethods[] = {
  {"update_neg_bin_ll_onestep", update_neg_bin_ll_onestep, METH_VARARGS, "docstring"},
  {"update_neg_bin_deriv_onestep", update_neg_bin_deriv_onestep, METH_VARARGS, "docstring"},
  {NULL, NULL, 0, NULL} /* Sentinel */
};


// Python module
static struct PyModuleDef fast_posterior = {
  PyModuleDef_HEAD_INIT,
  "fast_posterior",
  "A C++ wrapper of the Python posterior function.",
  -1,
  PosteriorMethods
};


PyMODINIT_FUNC
PyInit_fast_posterior(void){
  return PyModule_Create(&fast_posterior);
}


// From https://docs.python.org/3/extending/extending.html
int
main(int argc, char *argv[]) {
  wchar_t *program = Py_DecodeLocale(argv[0], NULL);
  if (program == NULL) {
      fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
      exit(1);
  }

  /* Add a built-in module, before Py_Initialize */
  PyImport_AppendInittab("fast_posterior", PyInit_fast_posterior);

  /* Pass argv[0] to the Python interpreter */
  Py_SetProgramName(program);

  /* Initialize the Python interpreter.  Required. */
  Py_Initialize();

  /* Optionally import the module; alternatively,
     import can be deferred until the embedded script
     imports it. */
  PyImport_ImportModule("fast_posterior");

  PyMem_RawFree(program);
  return 0;
}
