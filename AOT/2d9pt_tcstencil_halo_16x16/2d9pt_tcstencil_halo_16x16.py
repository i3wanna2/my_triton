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

"""

@triton.jit
def kernel(
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
    num_pid_m = tl.cdiv(M-2*R, BLOCK_SIZE_M) #一共有几个块
    num_pid_n = tl.cdiv(N-2*R, BLOCK_SIZE_N)
    #this是第几个块
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offset_16 = tl.arange(0, BLOCK_SIZE_M)

    #取parameter_hor
    para_h_ptrs = param_hor + ((offset_16[:,None])*BLOCK_SIZE_N + offset_16[None,:])
    para_h_data = tl.load(para_h_ptrs)

    para_v_ptrs = param_ver + ((offset_16[:,None])*BLOCK_SIZE_N + offset_16[None,:])
    para_v_data = tl.load(para_v_ptrs)

    offs_am = pid_m * BLOCK_SIZE_M + offset_16
    offs_an = pid_n * BLOCK_SIZE_N + offset_16

    a_ptrs = A + ((offs_am[:,None]+R)*N + offs_an[None,:]+R)
    a_data = tl.load(a_ptrs)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator = tl.dot(a_data, para_h_data, accumulator)
    accumulator = tl.dot(para_v_data, a_data, accumulator)

    #halo区 用scatter写
    halo_result = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    #右halo aptr+1
    halo_ptrs = a_ptrs + 1
    halo_data = tl.load(halo_ptrs, mask=(offset_16[None,:]==15))
    halo_result = halo_data+halo_result
    halo_data = tl.load(halo_ptrs, mask=(offset_16[None,:]==14))
    halo_result = halo_data+halo_result
    halo_ptrs = a_ptrs + 2
    halo_data = tl.load(halo_ptrs, mask=(offset_16[None,:]==15))
    halo_result = halo_data+halo_result

    #左halo aptr-1
    halo_ptrs = a_ptrs - 1
    halo_data = tl.load(halo_ptrs, mask=(offset_16[None,:]==0))
    halo_result = halo_data+halo_result
    halo_data = tl.load(halo_ptrs, mask=(offset_16[None,:]==1))
    halo_result = halo_data+halo_result
    halo_ptrs = a_ptrs - 2
    halo_data = tl.load(halo_ptrs, mask=(offset_16[None,:]==0))
    halo_result = halo_data+halo_result

    #上halo aptr-N
    halo_ptrs = a_ptrs - N
    halo_data = tl.load(halo_ptrs, mask=(offset_16[:,None]==0))
    halo_result = halo_data+halo_result
    halo_data = tl.load(halo_ptrs, mask=(offset_16[:,None]==1))
    halo_result = halo_data+halo_result
    halo_ptrs = a_ptrs - 2*N
    halo_data = tl.load(halo_ptrs, mask=(offset_16[:,None]==0))
    halo_result = halo_data+halo_result
    
    #下halo aptr-N
    halo_ptrs = a_ptrs + N
    halo_data = tl.load(halo_ptrs, mask=(offset_16[:,None]==14))
    halo_result = halo_data+halo_result
    halo_data = tl.load(halo_ptrs, mask=(offset_16[:,None]==15))
    halo_result = halo_data+halo_result
    halo_ptrs = a_ptrs + 2*N
    halo_data = tl.load(halo_ptrs, mask=(offset_16[:,None]==15))
    halo_result = halo_data+halo_result
    
    #累加并写回
    halo_result = accumulator + halo_result
    b_ptrs = B + ((offs_am[:,None]+R)*N + offs_an[None,:]+R)
    tl.store(b_ptrs, halo_result)


