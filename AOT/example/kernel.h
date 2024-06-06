#include <cuda.h>

CUresult kernel_520x520x2x16x16x12x12_warps1xstages3(CUstream stream, CUdeviceptr A, CUdeviceptr B, CUdeviceptr param_hor, CUdeviceptr param_ver);
void load_kernel_520x520x2x16x16x12x12_warps1xstages3();
void unload_kernel_520x520x2x16x16x12x12_warps1xstages3();
    
int kernel_get_num_algos(void);

CUresult kernel_default(CUstream stream, CUdeviceptr A, CUdeviceptr B, CUdeviceptr param_hor, CUdeviceptr param_ver);
CUresult kernel(CUstream stream, CUdeviceptr A, CUdeviceptr B, CUdeviceptr param_hor, CUdeviceptr param_ver, int algo_id);
void load_kernel();
void unload_kernel();
    