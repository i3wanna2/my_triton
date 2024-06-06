# gcc *.c -c -fPIC -I /usr/local/cuda/include
# gcc *.o -shared -o libstencil.so -L /usr/local/cuda/lib64
# gcc main.c -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -l cuda -L /root/my_triton/AOT -l stencil -o a.out