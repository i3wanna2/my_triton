#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "kernel.h"

int main()
{
    CUdevice dev;
    CUcontext ctx;
    CUstream stream;
    CUdeviceptr A, B, param_hor, param_ver;
    CUresult err = 0;
    cuInit(0);
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);

    cuMemAlloc(&A, 2 * (7200 + 16 * 2) * (7200 + 16 * 2));
    cuMemAlloc(&B, 2 * (7200 + 16 * 2) * (7200 + 16 * 2));
    cuMemAlloc(&param_hor, 2 * (16) * (16));
    cuMemAlloc(&param_ver, 2 * (16) * (16));

    cuStreamCreate(&stream, 0);
    load_kernel();

    cuStreamSynchronize(stream);
    kernel(stream, A, B, param_hor, param_ver, 0);
    unload_kernel();
    cuCtxDestroy(ctx);

    return 0;
}