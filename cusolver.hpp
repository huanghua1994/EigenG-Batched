#if defined(__NVCC__)

#include <cusolverDn.h>
#include "nvtx3/nvToolsExt.h"

#if 0
void
cusolver_test(int n, half *A, int lda, half *W, int batchSize)
{
}
#endif

float
cusolver_test(int n, double *A, int lda, double *W, int batchSize)
{
  cusolverDnHandle_t cusolverH = NULL;
  cusolverDnCreate(&cusolverH);
  cudaStream_t stream = NULL; 
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cusolverDnSetStream(cusolverH, stream);
  syevjInfo_t syevj_params = NULL;
  cusolverDnCreateSyevjInfo(&syevj_params);
  //Use default parameters
  //double constexpr EPS = (double)std::numeric_limits<double>::epsilon();
  //cusolverDnXsyevjSetTolerance(syevj_params,EPS*512);
  //cusolverDnXsyevjSetMaxSweeps(syevj_params,10);
  //cusolverDnXsyevjSetSortEig(syevj_params,0);

  cudaEvent_t start_event, stop_event;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);

  int lwork;
  cusolverDnDsyevjBatched_bufferSize(
    cusolverH,
    CUSOLVER_EIG_MODE_VECTOR,
    CUBLAS_FILL_MODE_LOWER,  //CUBLAS_FILL_MODE_UPPER,
    n,
    A,
    lda,
    W,
    &lwork,
    syevj_params,
    batchSize
  );

  double *d_work;
  cudaMalloc((void**)&d_work, sizeof(double)*lwork);
  int *d_info;
  cudaMalloc((void**)&d_info, sizeof(int)*batchSize);

  cudaEventRecord(start_event, stream);
  nvtxRangePushA("cusolverDnDsyevjBatched");
  cusolverDnDsyevjBatched(
    cusolverH,
    CUSOLVER_EIG_MODE_VECTOR,
    CUBLAS_FILL_MODE_LOWER,  //CUBLAS_FILL_MODE_UPPER,
    n,
    A,
    lda,
    W,
    d_work,
    lwork,
    d_info,
    syevj_params,
    batchSize
  );
  nvtxRangePop();
  cudaEventRecord(stop_event, stream);
  cudaEventSynchronize(start_event);
  cudaEventSynchronize(stop_event);
  float runtime_ms = 0;
  cudaEventElapsedTime(&runtime_ms, start_event, stop_event);

  cudaFree(d_work);
  //int * info = (int *)malloc(sizeof(int)*batchSize);
  //cudaMemcpy(info,d_info,sizeof(int)*batchSize,cudaMemcpyDeviceToHost);
  cudaFree(d_info);

  /*
  for(int i=0;i<batchSize;i++){
    if(info[i]!=0) {
      printf("[%06d] Gave up the iteration.\n",i); break;
    }
  }
  free(info);
  */

  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);

  cusolverDnDestroySyevjInfo(syevj_params);
  cusolverDnDestroy(cusolverH);
  cudaStreamDestroy(stream);
  return runtime_ms;
} 

float
cusolver_test(int n, float *A, int lda, float *W, int batchSize)
{
  cusolverDnHandle_t cusolverH = NULL;
  cusolverDnCreate(&cusolverH);
  cudaStream_t stream = NULL; 
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cusolverDnSetStream(cusolverH, stream);
  syevjInfo_t syevj_params = NULL;
  cusolverDnCreateSyevjInfo(&syevj_params);
  //Use default parameters
  //float constexpr EPS = std::numeric_limits<float>::epsilon();
  //cusolverDnXsyevjSetTolerance(syevj_params, EPS*16);
  //cusolverDnXsyevjSetMaxSweeps(syevj_params,10);
  //cusolverDnXsyevjSetSortEig(syevj_params,0);

  cudaEvent_t start_event, stop_event;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);

  int lwork;
  cusolverDnSsyevjBatched_bufferSize(
    cusolverH,
    CUSOLVER_EIG_MODE_VECTOR,
    CUBLAS_FILL_MODE_LOWER,
    n,
    A,
    lda,
    W,
    &lwork,
    syevj_params,
    batchSize
  );

  float *s_work;
  cudaMalloc((void**)&s_work, sizeof(float)*lwork);
  int *s_info;
  cudaMalloc((void**)&s_info, sizeof(int)*batchSize);

  cudaEventRecord(start_event, stream);
  nvtxRangePushA("cusolverDnSsyevjBatched");
  cusolverDnSsyevjBatched(
    cusolverH,
    CUSOLVER_EIG_MODE_VECTOR,
    CUBLAS_FILL_MODE_LOWER,
    n,
    A,
    lda,
    W,
    s_work,
    lwork,
    s_info,
    syevj_params,
    batchSize
  );
  nvtxRangePop();
  cudaEventRecord(stop_event, stream);
  cudaEventSynchronize(start_event);
  cudaEventSynchronize(stop_event);
  float runtime_ms = 0;
  cudaEventElapsedTime(&runtime_ms, start_event, stop_event);

  cudaFree(s_work);
  //int * info = (int *)malloc(sizeof(int)*batchSize);
  //cudaMemcpy(info,s_info,sizeof(int)*batchSize,cudaMemcpyDeviceToHost);
  cudaFree(s_info);

  /*
  for(int i=0;i<batchSize;i++){
    if(info[i]!=0) {
      printf("[%06d] Gave up the iteration.\n",i); break;
    }
  }
  free(info);
  */

  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);

  cusolverDnDestroySyevjInfo(syevj_params);
  cusolverDnDestroy(cusolverH);
  cudaStreamDestroy(stream);
  return runtime_ms;
} 

// This test requires CTK 12.6.2 or later
template<typename T>
float cusolver_ev_batch_test(int n_, T *A, int lda_, T *W, int batch_size_)
{
  int64_t n = (int64_t) n_;
  int64_t lda = (int64_t) lda_;
  int64_t batch_size = (int64_t) batch_size_;

  cusolverDnHandle_t handle = NULL;
  cusolverDnCreate(&handle);
  cudaStream_t stream = NULL;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cusolverDnSetStream(handle, stream);
  cusolverDnParams_t syev_params = NULL;
  cusolverDnCreateParams(&syev_params);
  cudaEvent_t start_event, stop_event;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);

  cudaDataType dtype = (sizeof(T) == 8) ? CUDA_R_64F : CUDA_R_32F;

  size_t host_work_bytes = 0, dev_work_bytes = 0;
  cusolverDnXsyevBatched_bufferSize(
    handle,
    syev_params,
    CUSOLVER_EIG_MODE_VECTOR,
    CUBLAS_FILL_MODE_LOWER,
    n,
    dtype,
    A,
    lda,
    dtype,
    W,
    dtype,
    &dev_work_bytes,
    &host_work_bytes,
    batch_size
  );

  void *dev_workspace = NULL, *host_workspace = NULL;
  int *dev_info = NULL;
  cudaMalloc(&dev_workspace, dev_work_bytes);
  cudaMallocHost(&host_workspace, host_work_bytes);
  cudaMalloc(&dev_info, sizeof(int) * batch_size);

  cudaEventRecord(start_event, stream);
  nvtxRangePushA("cusolverDnXsyevBatched");
  cusolverDnXsyevBatched(
    handle,
    syev_params,
    CUSOLVER_EIG_MODE_VECTOR,
    CUBLAS_FILL_MODE_LOWER,
    n,
    dtype,
    A,
    lda,
    dtype,
    W,
    dtype,
    dev_workspace,
    dev_work_bytes,
    host_workspace,
    host_work_bytes,
    dev_info,
    batch_size
  );
  nvtxRangePop();
  cudaEventRecord(stop_event, stream);
  cudaEventSynchronize(start_event);
  cudaEventSynchronize(stop_event);
  float runtime_ms = 0;
  cudaEventElapsedTime(&runtime_ms, start_event, stop_event);

  cudaFree(dev_workspace);
  cudaFreeHost(host_workspace);
  cudaFree(dev_info);

  cusolverDnDestroy(handle);
  cudaStreamDestroy(stream);
  cusolverDnDestroyParams(syev_params);
  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);
  return runtime_ms;
}

#endif

