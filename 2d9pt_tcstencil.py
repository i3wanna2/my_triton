import torch
import os

import triton
import triton.language as tl

"""

TCStencil简易版
没有换layout
写成dot就行, 分块大小可调
外部点分块大小: BLOCK_SIZE_M*BLOCK_SIZE_N
参数矩阵大小: BLOCK_SIZE_M*BLOCK_SIZE_N

"""

@triton.jit
def stencil_kernel(
    # 输入网格, 输出网格
    A, B,
    # A网格的大小
    M, N,
    #参数矩阵
    param_hor, param_ver,
    #半径大小
    R,
    #分块信息
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    INNER_SIZE_M: tl.constexpr,
    INNER_SIZE_N: tl.constexpr
):
    #把A存到B
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M-2*R, INNER_SIZE_M) #一共有几个块
    num_pid_n = tl.cdiv(N-2*R, INNER_SIZE_N)
    #this是第几个块
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    
    #取parameter_hor
    para_h_ptrs = param_hor + ((tl.arange(0, BLOCK_SIZE_M)[:,None])*BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None,:])
    para_h_data = tl.load(para_h_ptrs)

    para_v_ptrs = param_ver + ((tl.arange(0, BLOCK_SIZE_M)[:,None])*BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None,:])
    para_v_data = tl.load(para_v_ptrs)

    #load A， shape为[BLOCK_SIZE_M][BLOCK_SIZE_N]
    offs_bm = pid_m * INNER_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * INNER_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = A + ((offs_bm[:,None])*N + offs_bn[None,:])
    a_data = tl.load(a_ptrs)

    #乘加
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator = tl.dot(a_data, para_h_data, accumulator)
    accumulator = tl.dot(para_v_data, a_data, accumulator)

    b_ptrs = B + ((offs_bm[:,None])*N + offs_bn[None,:])
    t_bm = tl.arange(0, BLOCK_SIZE_M)
    t_bn = tl.arange(0, BLOCK_SIZE_N)

    tl.store(b_ptrs, accumulator, mask = (t_bm>=R)[:,None] & (t_bm<(BLOCK_SIZE_M-R))[:,None] & (t_bn>=R)[None,:] & (t_bn<(BLOCK_SIZE_N-R))[None,:])

def stencil(A, B, R, param_hor, param_ver, block_size_m, block_size_n):
    M, N = A.shape #输入网格大小
    grid = lambda META: (triton.cdiv(M-2*R, META['INNER_SIZE_M']) * triton.cdiv(N-2*R, META['INNER_SIZE_N']), )
    stencil_kernel[grid](
        A, B,
        M, N,
        param_hor, param_ver,
        R,
        BLOCK_SIZE_M = block_size_m, #外部点大小
        BLOCK_SIZE_N = block_size_n,
        INNER_SIZE_M = block_size_m-2*R, #内部点大小
        INNER_SIZE_N = block_size_n-2*R
    )
    return B

torch.manual_seed(0)
block_size_m=16
block_size_n=16
r = 2
m = 516 #内部点大小
n = 516
# a = torch.ones((m+2*r, n+2*r), device="cuda", dtype=torch.float16)
a = torch.randn((m+2*r, n+2*r), device="cuda", dtype=torch.float16)
b = torch.zeros((m+2*r, n+2*r), device=a.device, dtype=torch.float16)

param_hor = torch.zeros((block_size_m, block_size_n), device="cuda", dtype=torch.float16)
param_ver = torch.zeros((block_size_m, block_size_n), device="cuda", dtype=torch.float16)

#构造参数矩阵 hor
for j in range(0,block_size_n):
    if j>=r and (j+2*r+1-r)<=(block_size_m):
        for i in range(0,2*r+1):
            param_hor[j+i-r][j] = 1
# print(param_hor)

#构造参数矩阵 ver
for i in range(0,block_size_m):
    if i>=r and (i+2*r+1-r)<=(block_size_n):
        for j in range(0,2*r+1):
            if j==r:
                continue
            param_ver[i][i+j-r] = 1

# triton_output = stencil(a, b, r, param_hor, param_ver,block_size_m, block_size_n)

# gloden = torch.zeros((m+2*r, n+2*r), dtype=torch.float16)
# #gloden
# for i in range(r,m+r):
#     for j in range(r, n+r):
#         gloden[i][j] = a[i-1][j]+a[i-2][j]+a[i][j]+a[i+1][j]+a[i+2][j]+a[i][j-2]+a[i][j-1]+a[i][j+1]+a[i][j+2]

# print(gloden)
# print(triton_output)

quantiles = [0.5, 0.2, 0.8]
ms, min_ms, max_ms = triton.testing.do_bench(lambda: stencil(a, b, r, param_hor, param_ver,block_size_m, block_size_n), quantiles=quantiles)
print(ms)

# def benchmark(m,n,r):
#     A = torch.randn((m+2*r, n+2*r), device="cuda", dtype=torch.float16)
#     B = torch.empty((m+2*r, n+2*r), device=A.device, dtype=torch.float16)
#     quantiles = [0.5, 0.2, 0.8]
#     ms, min_ms, max_ms = triton.testing.do_bench(lambda: stencil(A, B, r), quantiles=quantiles)
#     return ms, max_ms, min_ms

# r = 2
# m = 512
# n = 512
# ms, max_ms, min_ms =benchmark(m,n,r)
# print(ms)