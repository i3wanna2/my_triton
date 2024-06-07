import torch
import os

import triton
import triton.language as tl

"""

TCStencil 部分优化
没有换layout

外部点分块大小定死: [BLOCK_SIZE_M==16]*[BLOCK_SIZE_N==16]
参数矩阵大小定死: [BLOCK_SIZE_M==12]*[BLOCK_SIZE_N==16]
每个block更新: [BLOCK_SIZE_M==16]*[BLOCK_SIZE_N==16]

A、B矩阵进行16*16存储, 边界点pad

"""

@triton.jit
def stencil_kernel(
    #输入网格, 输出网格
    A, B,
    #A网格的大小
    M, N,
    #参数矩阵
    param_hor, param_ver,
    #半径大小
    R,
    #分块信息
    BLOCK_SIZE_M: tl.constexpr, #==16
    BLOCK_SIZE_N: tl.constexpr #==16
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M-2*16, BLOCK_SIZE_M) #一共有几个块
    num_pid_n = tl.cdiv(N-2*16, BLOCK_SIZE_N)
    #this是第几个块（算上halo）
    pid_m = (pid // num_pid_n) + 1
    pid_n = (pid % num_pid_n) + 1

    offset_16 = tl.arange(0, BLOCK_SIZE_M)

    #取parameter_hor
    para_h_ptrs = param_hor + ((offset_16[:,None])*BLOCK_SIZE_N + offset_16[None,:])
    para_h_data = tl.load(para_h_ptrs)

    para_v_ptrs = param_ver + ((offset_16[:,None])*BLOCK_SIZE_N + offset_16[None,:])
    para_v_data = tl.load(para_v_ptrs)

    a_brick_offset = (pid_m*(N//16) + pid_n)*16*16

    a_ptrs = A + a_brick_offset + ((offset_16[:,None])*16 + offset_16[None,:])
    a_data = tl.load(a_ptrs)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator = tl.dot(a_data, para_h_data, accumulator)
    accumulator = tl.dot(para_v_data, a_data, accumulator)

    #halo区 用scatter写
    halo_result = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    #右halo aptr+1
    halo_ptrs = a_ptrs + (16-1)*16 + 2
    halo_data = tl.load(halo_ptrs, mask=(offset_16[None,:]==15))
    halo_result = halo_data+halo_result
    halo_data = tl.load(halo_ptrs, mask=(offset_16[None,:]==14))
    halo_result = halo_data+halo_result
    halo_ptrs = a_ptrs + (16-1)*16 + 1
    halo_data = tl.load(halo_ptrs, mask=(offset_16[None,:]==15))
    halo_result = halo_data+halo_result

    #左halo aptr-1
    halo_ptrs = a_ptrs - (16-1)*16 - 2
    halo_data = tl.load(halo_ptrs, mask=(offset_16[None,:]==0))
    halo_result = halo_data+halo_result
    halo_data = tl.load(halo_ptrs, mask=(offset_16[None,:]==1))
    halo_result = halo_data+halo_result
    halo_ptrs = a_ptrs - (16-1)*16 - 1
    halo_data = tl.load(halo_ptrs, mask=(offset_16[None,:]==0))
    halo_result = halo_data+halo_result

    #上halo aptr-N
    halo_ptrs = a_ptrs - (N//16-1)*16*16 - 2*16
    halo_data = tl.load(halo_ptrs, mask=(offset_16[:,None]==0))
    halo_result = halo_data+halo_result
    halo_data = tl.load(halo_ptrs, mask=(offset_16[:,None]==1))
    halo_result = halo_data+halo_result
    halo_ptrs = a_ptrs - (N//16-1)*16*16 - 1*16
    halo_data = tl.load(halo_ptrs, mask=(offset_16[:,None]==0))
    halo_result = halo_data+halo_result
    
    #下halo aptr-N
    halo_ptrs = a_ptrs + (N//16-1)*16*16 + 2*16
    halo_data = tl.load(halo_ptrs, mask=(offset_16[:,None]==14))
    halo_result = halo_data+halo_result
    halo_data = tl.load(halo_ptrs, mask=(offset_16[:,None]==15))
    halo_result = halo_data+halo_result
    halo_ptrs = a_ptrs + (N//16-1)*16*16 + 1*16
    halo_data = tl.load(halo_ptrs, mask=(offset_16[:,None]==15))
    halo_result = halo_data+halo_result
    
    #累加并写回
    halo_result = accumulator + halo_result
    b_ptrs = B + a_brick_offset + ((offset_16[:,None])*16 + offset_16[None,:])
    tl.store(b_ptrs, halo_result)


def stencil(A, B, R, param_hor, param_ver, block_size_m, block_size_n):
    M, N = A.shape #输入网格大小
    grid = lambda META: (triton.cdiv(M-2*R, META['BLOCK_SIZE_M']) * triton.cdiv(N-2*R, META['BLOCK_SIZE_M']), )
    stencil_kernel[grid](
        A, B,
        M, N,
        param_hor, param_ver,
        R,
        BLOCK_SIZE_M = block_size_m, #==16
        BLOCK_SIZE_N = block_size_n #==16
    )
    return B

torch.manual_seed(0)
block_size_m=16
block_size_n=16
r = 2
inner_m = 7200 #输出网格内部点大小
inner_n = 7200
#halo区pad 16*16块
a = torch.ones((inner_m+2*16, inner_n+2*16), device="cuda", dtype=torch.float16)
# a = torch.randn((inner_m+2*16, inner_n+2*16), device="cuda", dtype=torch.float16)
b = torch.zeros((inner_m+2*16, inner_n+2*16), device=a.device, dtype=torch.float16)

param_hor = torch.zeros((block_size_m, block_size_n), device="cuda", dtype=torch.float16)
param_ver = torch.zeros((block_size_m, block_size_n), device="cuda", dtype=torch.float16)

#构造参数矩阵 hor
for j in range(0,block_size_n):
    start_i = j-r if j>=r else 0
    end_i = start_i+2*r+1 if j>=r else r+1+j
    for i in range(start_i, end_i):
        if i>=block_size_m:
            break
        param_hor[i][j] = 1

# print(param_hor)

#构造参数矩阵 ver
for i in range(0,block_size_n):
    start_j = i-r if i>=r else 0
    end_j = start_j+2*r+1 if i>=r else r+1+i
    k = 0 if i>=r else r-i #不确定是否general
    for j in range(start_j, end_j):
        if k==r: #去中心点
            k = k+1
            continue
        if j>=block_size_n:
            break
        param_ver[i][j] = 1
        k = k+1

# print(param_ver)
# exit()

# triton_output = stencil(a, b, r, param_hor, param_ver,block_size_m, block_size_n)

# gloden = torch.zeros((m+2*r, n+2*r), dtype=torch.float16)
# #gloden
# for i in range(r,m+r):
#     for j in range(r, n+r):
#         gloden[i][j] = a[i-1][j]+a[i-2][j]+a[i][j]+a[i+1][j]+a[i+2][j]+a[i][j-2]+a[i][j-1]+a[i][j+1]+a[i][j+2]

# print(gloden)
# breakpoint()
# print(triton_output)

quantiles = [0.5, 0.2, 0.8]
ms, min_ms, max_ms = triton.testing.do_bench(lambda: stencil(a, b, r, param_hor, param_ver,block_size_m, block_size_n), quantiles=quantiles, warmup=50,rep=1000)
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