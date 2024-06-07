export LD_LIBRARY_PATH=/root/my_triton/AOT/2d9pt_tcstencil_halo_16x16
CUDA_VISIBLE_DEVICES=1 ncu --target-processes all -f -o profile ./test