import torch
import os

import triton
import triton.language as tl

"""

TCStencil 3D
换layout 16*16*16

每个block更新: [BLOCK_SIZE_M]*[BLOCK_SIZE_N]*[BLOCK_SIZE_K]

A、B矩阵进行16*16*16存储, 边界点pad

先stream, 每个平面load一个slice, 这个slice进行

3D分块, 每个block算一个brick [BLOCK_SIZE_M]*[BLOCK_SIZE_N]*[BLOCK_SIZE_K]
layout [BLOCK_SIZE_K]*[BLOCK_SIZE_M]*[BLOCK_SIZE_N]
"""

@triton.jit
def stencil_kernel(
    #输入网格, 输出网格
    A, B,
    #A网格的大小
    M, N, K,
    #参数矩阵
    param_hor, param_ver,
    #半径大小
    R,
    #分块信息
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    #这个要算的brick的id是多少
    pid_m = tl.program_id(axis=0) + 1
    pid_n = tl.program_id(axis=1) + 1
    pid_k = tl.program_id(axis=2) + 1

    #一共有几个brick
    num_brick_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_brick_n = tl.cdiv(M, BLOCK_SIZE_N)
    num_brick_k = tl.cdiv(M, BLOCK_SIZE_K)

    offset_1d_M = tl.arange(0, BLOCK_SIZE_M)
    offset_1d_N = tl.arange(0, BLOCK_SIZE_N)
    offset_1d_K = tl.arange(0, BLOCK_SIZE_K)
    slice_offsets = offset_1d_M[:,None]*BLOCK_SIZE_M + offset_1d_N[None,:]

    #前面有几个块，带来的偏移是多少，块的存储方式[K][M][N]，每个块[K][16*16]
    base_offset = (pid_k*num_brick_m*num_brick_n + pid_m*num_brick_n + pid_n)*BLOCK_SIZE_K*BLOCK_SIZE_M*BLOCK_SIZE_N

    #取parameter_hor
    para_h_ptrs = param_hor + slice_offsets
    para_h_data = tl.load(para_h_ptrs)

    para_v_ptrs = param_ver + slice_offsets
    para_v_data = tl.load(para_v_ptrs)

    #slice 存储结果
    buffer_0 = tl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_N),dtype=tl.float32)
    buffer_1 = tl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_N),dtype=tl.float32)
    buffer_2 = tl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_N),dtype=tl.float32)
    buffer_3 = tl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_N),dtype=tl.float32)
    buffer_4 = tl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_N),dtype=tl.float32)

    #延k扫描
    for k in range(-2, BLOCK_SIZE_K+2):
        if BLOCK_SIZE_K > k and k >= 0:
            #load slice A/B的存储顺序 [K][M][N] [K][16*16]
            k_offset = k * BLOCK_SIZE_M * BLOCK_SIZE_N
            a_slice_ptrs = A + base_offset + k_offset + slice_offsets
            a_slice_data = tl.load(a_slice_ptrs)

            #计算这个slice的tc stencil
            accumulator = tl.dot(a_slice_data, para_h_data)
            accumulator = tl.dot(para_v_data, a_slice_data, accumulator)
            
            #平面halo scatter
            halo_result = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            
            #右
            halo_ptrs = A + slice_offsets + (pid_k*num_brick_m*num_brick_n + pid_m*num_brick_n + (pid_n+1))*BLOCK_SIZE_K*BLOCK_SIZE_M*BLOCK_SIZE_N + k*BLOCK_SIZE_M*BLOCK_SIZE_N - (BLOCK_SIZE_M-2)
            halo_data = tl.load(halo_ptrs, mask=(offset_1d_N[None,:]>=(BLOCK_SIZE_M-2)))
            halo_result = halo_data+halo_result
            halo_data = tl.load(halo_ptrs-1, mask=(offset_1d_N[None,:]==(BLOCK_SIZE_M-1)))
            halo_result = halo_data+halo_result

            #左
            halo_ptrs = A + slice_offsets + (pid_k*num_brick_m*num_brick_n + pid_m*num_brick_n + (pid_n-1))*BLOCK_SIZE_K*BLOCK_SIZE_M*BLOCK_SIZE_N + k*BLOCK_SIZE_M*BLOCK_SIZE_N + (BLOCK_SIZE_M-2)
            halo_data = tl.load(halo_ptrs, mask=(offset_1d_N[None,:]<=1))
            halo_result = halo_data+halo_result
            halo_data = tl.load(halo_ptrs-1, mask=(offset_1d_N[None,:]==0))
            halo_result = halo_data+halo_result

            #上
            halo_ptrs = A + slice_offsets + (pid_k*num_brick_m*num_brick_n + (pid_m-1)*num_brick_n + pid_n)*BLOCK_SIZE_K*BLOCK_SIZE_M*BLOCK_SIZE_N + k*BLOCK_SIZE_M*BLOCK_SIZE_N + (BLOCK_SIZE_N-2)
            halo_data = tl.load(halo_ptrs, mask=(offset_1d_M[:,None]<=1))
            halo_result = halo_data+halo_result
            halo_data = tl.load(halo_ptrs+BLOCK_SIZE_M, mask=(offset_1d_M[:,None]==0))
            halo_result = halo_data+halo_result

            #下
            halo_ptrs = A + slice_offsets + (pid_k*num_brick_m*num_brick_n + (pid_m+1)*num_brick_n + pid_n)*BLOCK_SIZE_K*BLOCK_SIZE_M*BLOCK_SIZE_N + k*BLOCK_SIZE_M*BLOCK_SIZE_N - (BLOCK_SIZE_N-2)
            halo_data = tl.load(halo_ptrs, mask=(offset_1d_M[:,None]>=BLOCK_SIZE_M-2))
            halo_result = halo_data+halo_result
            halo_data = tl.load(halo_ptrs-BLOCK_SIZE_M, mask=(offset_1d_M[:,None]==BLOCK_SIZE_M-1))
            halo_result = halo_data+halo_result
            
            buffer_2 += halo_result
            buffer_2 += accumulator

        elif k < 0:
            #前一个的后几个
            a_slice_ptrs = A + slice_offsets + ((pid_k-1)*num_brick_m*num_brick_n + pid_m*num_brick_n + pid_n)*BLOCK_SIZE_K*BLOCK_SIZE_M*BLOCK_SIZE_N + (BLOCK_SIZE_K-k)*BLOCK_SIZE_M*BLOCK_SIZE_N
            a_slice_data = tl.load(a_slice_ptrs)
        else:
            a_slice_ptrs = A + slice_offsets + ((pid_k+1)*num_brick_m*num_brick_n + pid_m*num_brick_n + pid_n)*BLOCK_SIZE_K*BLOCK_SIZE_M*BLOCK_SIZE_N + (k-BLOCK_SIZE_K)*BLOCK_SIZE_M*BLOCK_SIZE_N
            a_slice_data = tl.load(a_slice_ptrs)

        if k>-2:
            buffer_0 += a_slice_data
        if k>-1:
            buffer_1 += a_slice_data
        if k<BLOCK_SIZE_K+2:
            buffer_3 += a_slice_data
        if k<BLOCK_SIZE_K+1:
            buffer_4 += a_slice_data
        
        #保存0的值
        if k>3:
            b_ptrs = B + base_offset + (k-2) * BLOCK_SIZE_M * BLOCK_SIZE_N + slice_offsets
            tl.store(b_ptrs,buffer_0)
        
        buffer_0 = buffer_1
        buffer_1 = buffer_2
        buffer_2 = buffer_3
        buffer_3 = buffer_4
        buffer_4 = tl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_N),dtype=tl.float32)


def stencil(A, B, R, param_hor, param_ver, block_size_m, block_size_n,block_size_k):
    M, N, K = A.shape #输入网格大小
    grid = lambda META: (triton.cdiv(M-2*block_size_m, META['BLOCK_SIZE_M']), triton.cdiv(N-2*block_size_n, META['BLOCK_SIZE_M']), triton.cdiv(N-2*block_size_k, META['BLOCK_SIZE_K']))
    stencil_kernel[grid](
        A, B,
        M, N, K,
        param_hor, param_ver,
        R,
        BLOCK_SIZE_M = block_size_m, #==16
        BLOCK_SIZE_N = block_size_n, #==16
        BLOCK_SIZE_K = block_size_k #16
    )
    return B

torch.manual_seed(0)
block_size_m=16
block_size_n=16
block_size_k=16
r = 2
inner_m = 1024 #输出网格内部点大小
inner_n = 1024
inner_k = 1024
#halo区pad 16*16块
a = torch.ones((inner_m+2*block_size_m, inner_n+2*block_size_m, inner_k+2*block_size_k), device="cuda", dtype=torch.float32)
# a = torch.randn((inner_m+2*16, inner_n+2*16), device="cuda", dtype=torch.float16)
b = torch.zeros((inner_m+2*block_size_m, inner_n+2*block_size_m, inner_k+2*block_size_k), device=a.device, dtype=torch.float32)

param_hor = torch.zeros((block_size_m, block_size_n), device="cuda", dtype=torch.float32)
param_ver = torch.zeros((block_size_m, block_size_n), device="cuda", dtype=torch.float32)

#构造参数矩阵 hor
for j in range(0,block_size_n):
    start_i = j-r if j>=r else 0
    end_i = start_i+2*r+1 if j>=r else r+1+j
    for i in range(start_i, end_i):
        if i>=block_size_m:
            break
        param_hor[i][j] = 1



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

# f = open("out","w")

# for i in range(0, block_size_m):
#     for j in range(0, block_size_n):
#         f.write(str(int(param_ver[i][j])))
#     f.write("\n")

# f.close()
# exit()

# triton_output = stencil(a, b, r, param_hor, param_ver, block_size_m, block_size_n)

# gloden = torch.zeros((m+2*r, n+2*r), dtype=torch.float16)
# #gloden
# for i in range(r,m+r):
#     for j in range(r, n+r):
#         gloden[i][j] = a[i-1][j]+a[i-2][j]+a[i][j]+a[i+1][j]+a[i+2][j]+a[i][j-2]+a[i][j-1]+a[i][j+1]+a[i][j+2]

# print(gloden)
# breakpoint()
# print(triton_output)

quantiles = [0.5, 0.2, 0.8]
ms, min_ms, max_ms = triton.testing.do_bench(lambda: stencil(a, b, r, param_hor, param_ver,block_size_m, block_size_n,block_size_k), quantiles=quantiles, warmup=500,rep=100)
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