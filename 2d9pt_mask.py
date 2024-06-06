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
mask版: 2d9pt
设inner大小为[a][b], 那么所需的数据load到[a][b][9]?中, 9个是这个点所需的全部
load时使用mask
然后reduce其它维, 得到[a][b]的输出

对比scatter, 仅改变了计算process, 分块方法一样
"""

@triton.jit
def stencil_kernel(
    # 输入网格, 输出网格
    A, B,
    # A网格的大小
    M, N,
    #半径大小
    R:tl.constexpr,
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
    pid_bm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    pid_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    #中心点 a中的位置
    offs_bm = pid_bm + R
    offs_bn = pid_bn + R
    radius_offs = tl.arange(0, 8)
    radius_offs_true = radius_offs - R

    #vertical tlie
    a_ptrs = A + ((offs_bm[:,None]+radius_offs_true[None,:])*N)[:,None,:] + offs_bn[None,:,None]
    # x, y+1
    a_data = tl.load(a_ptrs, mask = radius_offs[None,None,:]<5)
    result = tl.sum(a_data, axis=-1)
    # breakpoint()

    #horizontal tile
    a_ptrs = A + ((offs_bm[:,None]+radius_offs_true[None,:]))[:,None,:] + offs_bn[None,:,None]*N
    a_data = tl.load(a_ptrs, mask = radius_offs[None,None,:]<5)
    result += tl.sum(a_data, axis=-1)

    #减去中心点
    a_ptrs = A + (offs_bm)[:,None]*N + offs_bn[None,:]
    a_data = tl.load(a_ptrs)
    result -= a_data
    
    #写回
    b_ptrs = B + (offs_bm[:,None])*N + offs_bn[None,:]
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
# a = torch.ones((m+2*r, n+2*r), device="cuda", dtype=torch.float16)
# a = torch.randn((m+2*r, n+2*r), device="cuda", dtype=torch.float16)
# b = torch.empty((m+2*r, n+2*r), device=a.device, dtype=torch.float16)
# triton_output = stencil(a, b, r)
# print(a)
# print(triton_output)


def benchmark(m,n,r):
    A = torch.randn((m+2*r, n+2*r), device="cuda", dtype=torch.float16)
    B = torch.empty((m+2*r, n+2*r), device=A.device, dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: stencil(A, B, r), quantiles=quantiles)
    return ms, max_ms, min_ms

r = 2
m = 512 #内部点大小
n = 512
ms, max_ms, min_ms =benchmark(m,n,r)
print(ms)