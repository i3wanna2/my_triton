#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <mma.h>
#include "stencil.h"

int main(){
    CUdevice dev;
    CUcontext ctx;
    CUstream stream;
    CUdeviceptr A, B, param_hor, param_ver;
    CUresult err = 0;
    cuInit(0);
    cuDeviceGet(&dev, 1);
    cuCtxCreate(&ctx, 1, dev);
    
    cuMemAlloc(&A, sizeof(__half_raw) * (520) * (520));
    cuMemAlloc(&B, sizeof(__half_raw) * (520) * (520));
    cuMemAlloc(&param_hor, sizeof(__half_raw) * (16) * (16));
    cuMemAlloc(&param_ver, sizeof(__half_raw) * (16) * (16));

    
    cuStreamCreate(&stream, 0);
    load_stencil_kernel();

    // cuStreamSynchronize(stream);
    // stencil_kernel(stream, A, B, param_hor, param_ver, 0);

    return 0;
}