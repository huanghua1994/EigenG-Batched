#if defined(__NVCC__)

#include <cusolverDn.h>
#include "nvtx3/nvToolsExt.h"

template<typename T>
float cusolver_test(int n, T *A, int lda, T *W, int batchSize, const bool check_ret_info = false)
{
  cusolverDnHandle_t cusolverH = NULL;
  cusolverDnCreate(&cusolverH);
  cudaStream_t stream = NULL; 
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cusolverDnSetStream(cusolverH, stream);
  syevjInfo_t syevj_params = NULL;
  cusolverDnCreateSyevjInfo(&syevj_params);
  //Use default parameters
  //T constexpr EPS = std::numeric_limits<T>::epsilon();
  //cusolverDnXsyevjSetTolerance(syevj_params, EPS*16);
  //cusolverDnXsyevjSetMaxSweeps(syevj_params,10);
  //cusolverDnXsyevjSetSortEig(syevj_params,0);

  cusolverStatus_t status;
  cudaError_t err;

  cudaEvent_t start_event, stop_event;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);

  int lwork;
  if (sizeof(T) == 8)
  {
    status = cusolverDnDsyevjBatched_bufferSize(
      cusolverH,
      CUSOLVER_EIG_MODE_VECTOR,
      CUBLAS_FILL_MODE_LOWER,
      n,
      reinterpret_cast<double*>(A),
      lda,
      reinterpret_cast<double*>(W),
      &lwork,
      syevj_params,
      batchSize
    );
  }
  if (sizeof(T) == 4)
  {
    status = cusolverDnSsyevjBatched_bufferSize(
      cusolverH,
      CUSOLVER_EIG_MODE_VECTOR,
      CUBLAS_FILL_MODE_LOWER,
      n,
      reinterpret_cast<float*>(A),
      lda,
      reinterpret_cast<float*>(W),
      &lwork,
      syevj_params,
      batchSize
    );
  }
  if (status != CUSOLVER_STATUS_SUCCESS)
  {
    printf("[ERROR] cusolverDnXsyevjBatched_bufferSize failed with error code %d\n", status);
    return 0.0f;
  }

  T *d_work;
  err = cudaMalloc((void**)&d_work, sizeof(T)*lwork);
  if (err != cudaSuccess)
  {
    printf("[ERROR] cudaMalloc for size %zu failed with code %d\n", sizeof(T)*lwork, err);
    return 0.0f;
  }
  int *d_info;
  err = cudaMalloc((void**)&d_info, sizeof(int)*batchSize);
  if (err != cudaSuccess)
  {
    printf("[ERROR] cudaMalloc for size %zu failed with code %d\n", sizeof(int)*batchSize, err);
    return 0.0f;
  }

  cudaEventRecord(start_event, stream);
  if (sizeof(T) == 8)
  {
    nvtxRangePushA("cusolverDnDsyevjBatched");
    status = cusolverDnDsyevjBatched(
      cusolverH,
      CUSOLVER_EIG_MODE_VECTOR,
      CUBLAS_FILL_MODE_LOWER,
      n,
      reinterpret_cast<double*>(A),
      lda,
      reinterpret_cast<double*>(W),
      reinterpret_cast<double*>(d_work),
      lwork,
      d_info,
      syevj_params,
      batchSize
    );
  }
  if (sizeof(T) == 4)
  {
    nvtxRangePushA("cusolverDnSsyevjBatched");
    status = cusolverDnSsyevjBatched(
      cusolverH,
      CUSOLVER_EIG_MODE_VECTOR,
      CUBLAS_FILL_MODE_LOWER,
      n,
      reinterpret_cast<float*>(A),
      lda,
      reinterpret_cast<float*>(W),
      reinterpret_cast<float*>(d_work),
      lwork,
      d_info,
      syevj_params,
      batchSize
    );
  }
  nvtxRangePop();
  cudaEventRecord(stop_event, stream);
  cudaEventSynchronize(start_event);
  cudaEventSynchronize(stop_event);
  float runtime_ms = 0;
  cudaEventElapsedTime(&runtime_ms, start_event, stop_event);
  if (status != CUSOLVER_STATUS_SUCCESS)
  {
    printf("[ERROR] cusolverDnXsyevjBatched failed with error code %d\n", status);
    return 0.0f;
  }

  if (check_ret_info)
  {
    int * info = (int *)malloc(sizeof(int)*batchSize);
    cudaMemcpy(info,d_info,sizeof(int)*batchSize,cudaMemcpyDeviceToHost);
    for(int i=0;i<batchSize;i++){
      if(info[i]!=0) {
        printf("XsyevjBatched failed for %d-th problem\n",i); break;
      }
    }
    free(info);
  }
 
  cudaFree(d_work);
  cudaFree(d_info);

  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);

  cusolverDnDestroySyevjInfo(syevj_params);
  cusolverDnDestroy(cusolverH);
  cudaStreamDestroy(stream);
  return runtime_ms;
} 

// This test requires CTK 12.6.2 or later
template<typename T>
float cusolver_ev_batch_test(int n_, T *A, int lda_, T *W, int batch_size_, const bool check_ret_info = false)
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

  cusolverStatus_t status;
  cudaError_t err;

  size_t host_work_bytes = 0, dev_work_bytes = 0;
  status = cusolverDnXsyevBatched_bufferSize(
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
  if (status != CUSOLVER_STATUS_SUCCESS)
  {
    printf("[ERROR] cusolverDnXsyevBatched_bufferSize failed with error code %d\n", status);
    return 0.0f;
  }

  void *dev_workspace = NULL, *host_workspace = NULL;
  int *dev_info = NULL;
  err = cudaMalloc(&dev_workspace, dev_work_bytes);
  if (err != cudaSuccess)
  {
    printf("[ERROR] cudaMalloc for size %zu failed with code %d\n", dev_work_bytes, err);
    return 0.0f;
  }
  err = cudaMallocHost(&host_workspace, host_work_bytes);
  if (err != cudaSuccess)
  {
    printf("[ERROR] cudaMallocHost for size %zu failed with code %d\n", host_work_bytes, err);
    return 0.0f;
  }
  err = cudaMalloc(&dev_info, sizeof(int) * batch_size);
  if (err != cudaSuccess)
  {
    printf("[ERROR] cudaMalloc for size %zu failed with code %d\n", sizeof(int) * batch_size, err);
    return 0.0f;
  }

  cudaEventRecord(start_event, stream);
  nvtxRangePushA("cusolverDnXsyevBatched");
  status = cusolverDnXsyevBatched(
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

  if (status != CUSOLVER_STATUS_SUCCESS)
  {
    printf("[ERROR] cusolverDnXsyevBatched failed with error code %d\n", status);
    return 0.0f;
  }

  if (check_ret_info)
  {
    int * info = (int *)malloc(sizeof(int)*batch_size_);
    cudaMemcpy(info,dev_info,sizeof(int)*batch_size_,cudaMemcpyDeviceToHost);
    for(int i=0;i<batch_size_;i++){
      if(info[i]!=0) {
        printf("XsyevBatched failed for %d-th problem\n",i); break;
      }
    }
    free(info);
  }

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

