#include <cuda.h>
#include <stdint.h>
#include <assert.h>

// launcher for: kernel_520x520x2x16x16x12x12_warps1xstages3
CUresult kernel_1d5e97e7_0123(CUstream stream, CUdeviceptr A, CUdeviceptr B, CUdeviceptr param_hor, CUdeviceptr param_ver);

CUresult kernel_520x520x2x16x16x12x12_warps1xstages3(CUstream stream, CUdeviceptr A, CUdeviceptr B, CUdeviceptr param_hor, CUdeviceptr param_ver){
if (1)
    return kernel_1d5e97e7_0123(stream, A, B, param_hor, param_ver);

  return CUDA_ERROR_INVALID_VALUE;
}

// load for: kernel_520x520x2x16x16x12x12_warps1xstages3
void load_kernel_1d5e97e7_0123();
void load_kernel_520x520x2x16x16x12x12_warps1xstages3() {
  load_kernel_1d5e97e7_0123();
}

// unload for: kernel_520x520x2x16x16x12x12_warps1xstages3
void unload_kernel_1d5e97e7_0123();
void unload_kernel_520x520x2x16x16x12x12_warps1xstages3() {
  unload_kernel_1d5e97e7_0123();
}

typedef CUresult (*kernel_func_t)(CUstream stream, CUdeviceptr A, CUdeviceptr B, CUdeviceptr param_hor, CUdeviceptr param_ver);
kernel_func_t kernel_kernels[] = {
  kernel_520x520x2x16x16x12x12_warps1xstages3,
};

int kernel_get_num_algos(void){
  return (int)(sizeof(kernel_kernels) / sizeof(kernel_kernels[0]));
}

CUresult kernel(CUstream stream, CUdeviceptr A, CUdeviceptr B, CUdeviceptr param_hor, CUdeviceptr param_ver, int algo_id){
  assert (algo_id < (int)sizeof(kernel_kernels));
  return kernel_kernels[algo_id](stream, A, B, param_hor, param_ver);
}

void load_kernel(void){
  load_kernel_520x520x2x16x16x12x12_warps1xstages3();
}

void unload_kernel(void){
  unload_kernel_520x520x2x16x16x12x12_warps1xstages3();
}


CUresult kernel_default(CUstream stream, CUdeviceptr A, CUdeviceptr B, CUdeviceptr param_hor, CUdeviceptr param_ver){
  return kernel(stream, A, B, param_hor, param_ver, 0);
}
