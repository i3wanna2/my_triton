export LD_LIBRARY_PATH=/root/my_triton/AOT/2d9pt_tcstencil_layout
# CUDA_VISIBLE_DEVICES=1 ncu --clock-control none --target-processes all --set full -f -o profile_7200 ./test
CUDA_VISIBLE_DEVICES=1 ncu --clock-control none --target-processes all --set full -f -o profile_7200 ./test