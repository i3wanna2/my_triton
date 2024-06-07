import torch
import os

import triton
import triton.language as tl

"""

每个block负责更新 BLOCK_SIZE_M*BLOCK_SIZE_N 个元素
每个线程更新一个点
以inner点为分块依据
A、B点一样大
注意: inner一定整除, 没写非整除逻辑

"""

@triton.jit
def stencil_kernel(
    # 输入网格, 输出网格
    A, B,
    # A网格的大小
    M, N,
    #半径大小
    R,
    #分块信息
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    #把A存到B
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M-2*R, BLOCK_SIZE_M) #一共有几个块
    num_pid_n = tl.cdiv(N-2*R, BLOCK_SIZE_N)
    #this是第几个块
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    #每个块负责 BLOCK_SIZE_M*BLOCK_SIZE_N 个元素
    #this的块右上角元素 B的
    offs_bm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_an = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)


    #计算
    result = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)

    #竖reduce
    a_ptrs = A + ((offs_am[:,None])*N + offs_an[None,:]+R) 
    a_data = tl.load(a_ptrs)
    result = result + a_data

    a_ptrs = A + (((offs_am[:,None])+1)*N + offs_an[None,:]+R) 
    a_data = tl.load(a_ptrs)
    result = result + a_data

    a_ptrs = A + (((offs_am[:,None])+2)*N + offs_an[None,:]+R) 
    a_data = tl.load(a_ptrs)
    result = result + a_data

    a_ptrs = A + (((offs_am[:,None])+3)*N + offs_an[None,:]+R) 
    a_data = tl.load(a_ptrs)
    result = result + a_data

    a_ptrs = A + (((offs_am[:,None])+4)*N + offs_an[None,:]+R) 
    a_data = tl.load(a_ptrs)
    result = result + a_data

    #横reduce
    a_ptrs = A + (((offs_am[:,None])+R)*N + offs_an[None,:] + 0) 
    a_data = tl.load(a_ptrs)
    result = result + a_data

    a_ptrs = A + (((offs_am[:,None])+R)*N + offs_an[None,:] + 1) 
    a_data = tl.load(a_ptrs)
    result = result + a_data

    a_ptrs = A + (((offs_am[:,None])+R)*N + offs_an[None,:] + 3) 
    a_data = tl.load(a_ptrs)
    result = result + a_data

    a_ptrs = A + (((offs_am[:,None])+R)*N + offs_an[None,:] + 4) 
    a_data = tl.load(a_ptrs)
    result = result + a_data
    
    b_ptrs = B + ((offs_bm[:,None]+R)*N + offs_bn[None,:] + R)
    tl.store(b_ptrs, result)

def stencil(A, B, R):
    M, N = A.shape
    grid = lambda META: (triton.cdiv(M-2*R, META['BLOCK_SIZE_M']) * triton.cdiv(N-2*R, META['BLOCK_SIZE_N']), )
    stencil_kernel[grid](
        A, B,
        M, N,
        R,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 16
    )
    return B

# torch.manual_seed(0)
# r = 2
# m = 4
# n = 4
# # a = torch.ones((m+2*r, n+2*r), device="cuda", dtype=torch.float16)
# a = torch.randn((m+2*r, n+2*r), device="cuda", dtype=torch.float16)
# triton_output = stencil(a, r)
# print(a)
# print(triton_output)


def benchmark(m,n,r):
    A = torch.randn((m+2*r, n+2*r), device="cuda", dtype=torch.float16)
    B = torch.empty((m+2*r, n+2*r), device=A.device, dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: stencil(A, B, r), quantiles=quantiles,warmup=500,rep=5000)
    return ms, max_ms, min_ms

r = 2
m = 7200
n = 7200
ms, max_ms, min_ms =benchmark(m,n,r)
print(ms)