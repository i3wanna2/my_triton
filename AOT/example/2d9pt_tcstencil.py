import torch
import os

import triton
import triton.language as tl

@triton.jit
def kernel(
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
