#include <stdio.h>
#include <math.h>
#include <limits>
#include <omp.h>
#include <vector>

#if defined(__NVCC__)
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

#include "gpu_arch.h"
#include "misc.hpp"
#include "eigen_GPU_batch.hpp"
#include "eigen_GPU_check.hpp"

#if defined(__NVCC__)
#include "cusolver.hpp"
#include "cusolver_evd.hpp"
#endif
#if defined(__HIPCC__)
#include "hipsolver.hpp"
#endif


#if 0
extern "C" {
// for calling the original fortran EISPACK routines
  void tred1_(int *, int *, double *, double *, double *);
  void tql2_(int *, int *, double *, double *, double *, int *);
  void trbak1_(int *, int *, double *, int *, double *);
}
#endif

enum class Matrix_type {
  MATRIX_SYM_RAND,
  MATRIX_LETKF,
  MATRIX_FRANK
};
enum class Solver_type {
  EIGENG_BATCH,
  CUSOLVER_EVJ_BATCH,
  CUSOLVER_EV_BATCH,
  CUSOLVER_EVD,
  HIPSOLVER_EVJ_BATCH,
  HIPSOLVER_EVD
};


template < class T > __host__ T
rand_R( unsigned int * seed_ptr )
{
  return (T)( ((double)rand_r(seed_ptr))/RAND_MAX );
}

template < class T > __host__ void
set_mat(T *a, const int nm, const int n, const Matrix_type type, const int seed_)
{
  unsigned int seed = seed_;

  if ( type == Matrix_type::MATRIX_LETKF ) {

    T * w = (T *)malloc(sizeof(T)*n);
    for(int i=1;i<=n;i++){
      const T eps = std::numeric_limits<T>::epsilon();
      const T R = 16-1;
      const T delta = R*(2*rand_R<T>(&seed)-1.)*eps;
      const T DELTA = 1.0+delta;
      w[i-1] = (T)(n-1)*DELTA;
    }
    const int numV=2;
    for(int i=1;i<=numV;i++){
      const int j=(rand_r(&seed)%n);
      w[j] += rand_R<T>(&seed)*n;
    }

    T scale = 0;
#pragma omp parallel for
    for(int i=1;i<=n;i++){
      scale = (T)fmax(scale, fabs(w[i-1]));
    }
    scale = (T)fmax(scale, 1.);
#pragma omp parallel for
    for(int i=1;i<=n;i++){
      w[i-1] /= scale;
    }

    T * ht = (T *)malloc(sizeof(T)*n);
#pragma omp parallel for
    for(int i=1;i<=n;i++){
      ht[i-1] = (T)sqrt((double)i);
    }

#pragma omp parallel
    {

    T * hi = (T *)malloc(sizeof(T)*n);
    T * hj = (T *)malloc(sizeof(T)*n);

    int i0=0, j0=0;
#pragma omp for collapse(2)
    for(int i=1;i<=n;i++){
    for(int j=1;j<=n;j++){

    if(i0!=i){
    if(i==1){
      T hi_ = ht[n-1];
#pragma omp simd
      for(int k=1;k<=n;k++) hi[k-1] = 1. / hi_;
    } else {
      T s = (T)(i-1);
      T hi_ = ht[i-2] * ht[i-1];
#pragma omp simd
      for(int k=1;k<=i-1;k++) hi[k-1] = 1. / hi_;
      hi[i-1] = -s / hi_;
#pragma omp simd
      for(int k=i+1;k<=n;k++) hi[k-1] = 0.;
    }
    i0=i;
    }

    if(j0!=j){
    if(j==1){
      T hj_ = ht[n-1];
#pragma omp simd
      for(int k=1;k<=n;k++) hj[k-1] = 1. / hj_;
    } else {
      T s = (T)(j-1);
      T hj_ = ht[j-2] * ht[j-1];
#pragma omp simd
      for(int k=1;k<=j-1;k++) hj[k-1] = 1. / hj_;
      hj[j-1] = -s / hj_;
#pragma omp simd
      for(int k=j+1;k<=n;k++) hj[k-1] = 0.;
    }
    j0=j;
    }

      T t = 0.;
#pragma omp simd
      for(int k=1;k<=n;k++) {
        const T hijk = hi[k-1]*hj[k-1];
        t += w[k-1]*hijk;
      }
      a[(j-1)+(i-1)*nm] = t;

    }
    }
      free(hi);
      free(hj);
    }

#pragma omp parallel for collapse(2)
    for(int i=1;i<=n;i++){
    for(int j=1;j<=n;j++){
      a[(j-1)+(i-1)*nm] *= scale;
    }}

    free(w); free(ht);

  } else {

#pragma omp parallel for
    for(int i=0;i<n;i++) {
    for(int j=0;j<=i;j++) {
      T x, t;
      switch ( type ) {
      case Matrix_type::MATRIX_SYM_RAND:
#pragma omp critical
	{
        t = static_cast<T>(static_cast<double>(rand_r(&seed))/RAND_MAX);
	}
        x = 2*t - static_cast<T>(1.0);
        break;
      case Matrix_type::MATRIX_FRANK:
	{
        int k = min(j+1,i+1);
        x = static_cast<T>(k);
	}
        break;
      default:
        x = static_cast<T>(0.);
        break;
      }
      a[i+nm*j] = x; a[j+nm*i] = x;
    }}
  }
}

template < class T, Solver_type Solver > __host__ float
GPU_batch_test(
  const int num_iter, const int L, const int n, const Matrix_type type, 
  const bool accuracy_test = false, const bool print_timing = false
)
{
  const int nm = n;
  const int m  = n;
  size_t len;

  len = sizeof(T)*L*nm*n;
  T *a_h = NULL, *a_d = NULL, *b_d = NULL, *w_d = NULL;
  a_h = (T *)malloc(len);
  gpuMalloc(&a_d, len);
  gpuMalloc(&b_d, len);
  len = sizeof(T)*L*m;
  gpuMalloc(&w_d, len);
  if (a_h == NULL || a_d == NULL || b_d == NULL || w_d == NULL)
  {
    printf("Failed to allocate test matrices\n");
    free(a_h);
    gpuFree(a_d);
    gpuFree(b_d);
    gpuFree(w_d);
    return -1.0f;
  }

  gpuDeviceSynchronize();
  double ts = get_wtime();
  #pragma omp parallel for
  for(int id=0; id<L; id++) {
    T *a = a_h + (size_t)id*(nm*n);
    set_mat(a, nm, n, type, id);
  }
  gpuDeviceSynchronize();
  double te = get_wtime();
  if (print_timing) printf("  Data generation :: %le [s]\n", te-ts);

  len = sizeof(T)*L*nm*n;
  gpuDeviceSynchronize();
  ts = get_wtime();
  gpuMemcpy(b_d, a_h, len, gpuMemcpyHostToDevice);
  te = get_wtime();
  if (print_timing) printf("  Host -> Dev :: %le [s]\n", te-ts);

  T *wk_d = NULL;
  if ( Solver == Solver_type::EIGENG_BATCH ) {
    eigen_GPU_batch_BufferSize(L, nm, n, m, a_d, w_d, &len);
    gpuMalloc(&wk_d, len);
    if ( wk_d == NULL ) { 
      printf("Failed to allocate work array for EigenG-Batch\n");
      free(a_h);
      gpuFree(a_d); 
      gpuFree(b_d); 
      gpuFree(w_d); 
      return -1.0f;
    }
  }

  gpuStream_t stream;
  gpuStreamCreate( &stream );

  double total_time = 0;
  for (int i_test = 0; i_test < num_iter; i_test++) 
  {
    len = sizeof(T)*L*nm*n;
    gpuMemcpy(a_d, b_d, len, gpuMemcpyDeviceToDevice);

    float runtime_ms = 0;
    switch ( Solver )  {
    case Solver_type::EIGENG_BATCH:
      runtime_ms = eigen_GPU_batch(L, nm, n, m, a_d, w_d, wk_d, stream);
      break;
    #if defined(__NVCC__)
    case Solver_type::CUSOLVER_EVJ_BATCH:
      runtime_ms = cusolver_test<T>(n, a_d, nm, w_d, L);
      break;
    case Solver_type::CUSOLVER_EV_BATCH:
      runtime_ms = cusolver_ev_batch_test<T>(n, a_d, nm, w_d, L);
      break;
    case Solver_type::CUSOLVER_EVD:
      runtime_ms = cusolver_evd_test(n, a_d, nm, w_d, L);
      break;
    #endif
    #if defined(__HIPCC__)
    case Solver_type::HIPSOLVER_EVJ_BATCH:
      hipsolver_test(n, a_d, nm, w_d, L);
      break;
    #endif
    default:
      break;
    }
    if (runtime_ms < 0.0f) return -1.0f;
    total_time += (double) runtime_ms * 1e-3;
  }

  if (accuracy_test) 
  {
    gpuDeviceSynchronize();
    ts = get_wtime();
    eigen_GPU_check(L, nm, n, m, b_d, w_d, a_d, stream);
    te = get_wtime();
    if (print_timing) printf("  Accuracy Test :: %le [s]\n", te-ts);
  }

  gpuStreamDestroy( stream );
  double tm = total_time / num_iter;
  double flop = L*(double)n*n*n*(4./3+2.);
  double ld_data = sizeof(T)*L*(double)n*(n/2.    +2.   +n/2.+n);
  double st_data = sizeof(T)*L*(double)n*(n/2.+2. +n+1. +n     );
  double data = ld_data + st_data;

  printf("%4d, %.4le, %.4le, %.4le\n", n, tm, 1e-9*flop/tm, 1e-9*data/tm);
  fflush(stdout);

  gpuFree(a_d);
  gpuFree(b_d);
  gpuFree(w_d);
  if ( Solver == Solver_type::EIGENG_BATCH ) { gpuFree(wk_d); }
  free(a_h);
  return (float) tm;
}

template <class T, Solver_type Solver> __host__ 
void GPU_batches_test(
  const char *test_name, const int num_iter, const int batch_size, const int *mat_sizes, 
  const Matrix_type type, const bool accuracy_test = false, const bool print_timing = false
)
{
  printf("%s, %d iterations\n", test_name, num_iter);
  std::vector<float> timers;
  printf("N, Runtime (s), GF/s, GB/s\n");
  for (int i = 0; mat_sizes[i] > 0; i++)
  {
    float runtime = GPU_batch_test<T, Solver>(num_iter, batch_size, mat_sizes[i], type, accuracy_test, print_timing);
    if (runtime < 0.0f) break;
    timers.push_back(runtime);
  }
  printf("\nRuntime only:\n");
  for (int i = 0; i < timers.size(); i++) printf("%.4le\n", timers[i]);
  timers.clear();
}

int main(int argc, char* argv[])
{
  print_header("GPU-Batch-eigensolver", argc, argv);
  gpuSetDevice(0);

  const int num_iter = 10;
  //  const Matrix_type type = Matrix_type::MATRIX_FRANK;
  //  const Matrix_type type = Matrix_type::MATRIX_LETKF;
  const Matrix_type type = Matrix_type::MATRIX_SYM_RAND;

  int batch_size = 512;
  bool test_fp32 = true;
  bool test_fp64 = true;
  bool test_eigeng_batch = true;
  bool test_cu_evj_batch = true;
  bool test_cu_ev_batch = true;
  bool test_cu_evd_repeat = false;
  bool print_timing = false;
  bool accuracy_test = false;
  if (argc >= 2) batch_size = atoi(argv[1]);
  if (argc >= 3) test_fp32 = atoi(argv[2]) > 0;
  if (argc >= 4) test_fp64 = atoi(argv[3]) > 0;
  if (argc >= 5) test_eigeng_batch = atoi(argv[4]) > 0;
  if (argc >= 6) test_cu_evj_batch = atoi(argv[5]) > 0;
  if (argc >= 7) test_cu_ev_batch = atoi(argv[6]) > 0;
  if (argc >= 8) test_cu_evd_repeat = atoi(argv[7]) > 0;
  if (argc >= 9) print_timing = atoi(argv[8]) > 0;
  if (argc >= 10) accuracy_test = atoi(argv[9]) > 0;
  printf("\n $$$$$ Test settings $$$$$\n");
  printf("  <batch-size>                : %d\n", batch_size);
  printf("  <test-fp32>                 : %d\n", test_fp32);
  printf("  <test-fp64>                 : %d\n", test_fp64);
  printf("  <test-EigenG-batch>         : %d\n", test_eigeng_batch);
  printf("  <test-cusolver-evj-batch>   : %d\n", test_cu_evj_batch);
  printf("  <test-cusolver-ev-batch>    : %d\n", test_cu_ev_batch);
  printf("  <test-cusolver-evd-repeat>  : %d\n", test_cu_evd_repeat);


  #if defined(__NVCC__)
  const int nums[] = { 4, 8, 16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 0 };
  #endif
  #if defined(__HIPCC__)
  //const int nums[] = { 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 20, 23, 24, 25, 28, 31, 32, 33, 47, 48, 49, 63, 64, 65, 95, 96, 97, 127, 128, 129, 159, 160, 161, 191, 192, 193, 223, 224, 225, 255, 256, 0 };
  const int nums[] = { 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 20, 23, 24, 25, 28, 31, 32, 33, 47, 48, 49, 63, 64, 65, 95, 96, 97, 127, 128, 129, 0 };
  #endif

  if (test_eigeng_batch)
  {
    if (test_fp32) GPU_batches_test<float, Solver_type::EIGENG_BATCH>("EigenG-batch, FP32", num_iter, batch_size, nums, type, accuracy_test, print_timing);
    if (test_fp64) GPU_batches_test<double, Solver_type::EIGENG_BATCH>("EigenG-batch, FP64", num_iter, batch_size, nums, type, accuracy_test, print_timing);
  }

  #if defined(__NVCC__)
  if (test_cu_evj_batch)
  {
    if (test_fp32) GPU_batches_test<float, Solver_type::CUSOLVER_EVJ_BATCH>("cuSolver-SsyevjBatch", num_iter, batch_size, nums, type, accuracy_test, print_timing);
    if (test_fp64) GPU_batches_test<double, Solver_type::CUSOLVER_EVJ_BATCH>("cuSolver-DsyevjBatch", num_iter, batch_size, nums, type, accuracy_test, print_timing);
  }
  if (test_cu_ev_batch)
  {
    if (test_fp32) GPU_batches_test<float, Solver_type::CUSOLVER_EV_BATCH>("cuSolver-SsyevBatch", num_iter, batch_size, nums, type, accuracy_test, print_timing);
    if (test_fp64) GPU_batches_test<double, Solver_type::CUSOLVER_EV_BATCH>("cuSolver-DsyevBatch", num_iter, batch_size, nums, type, accuracy_test, print_timing);
  }
  if (test_cu_evd_repeat)
  {
    if (test_fp32) GPU_batches_test<float, Solver_type::CUSOLVER_EVD>("cuSolver-Ssyevd", num_iter, batch_size, nums, type, accuracy_test, print_timing);
    if (test_fp64) GPU_batches_test<double, Solver_type::CUSOLVER_EVD>("cuSolver-Dsyevd", num_iter, batch_size, nums, type, accuracy_test, print_timing);
  }
  #endif

  #if defined(__HIPCC__)
  printf("\n");
  printf(">> float hipsolver Jacobi acc_chk and avarage of the %d iterations.\n",iter);
  for(int n=8; n<=64; n*=2) {
    GPU_batch_test<float,Solver_type::HIPSOLVER_EVJ_BATCH>(1, 1, n, type, true);
    GPU_batch_test<float,Solver_type::HIPSOLVER_EVJ_BATCH>(iter, numBatch, n, type, false);
  }
  printf("\n");
  printf(">> double hipsolver Jacobi acc_chk and avarage out of the %d iterations.\n",iter);
  for(int n=8; n<=64; n*=2) {
    GPU_batch_test<double,Solver_type::HIPSOLVER_EVJ_BATCH>(1, 1, n, type, true);
    GPU_batch_test<double,Solver_type::HIPSOLVER_EVJ_BATCH>(iter, numBatch, n, type, false);
  }
  #endif

  return 0;
}


