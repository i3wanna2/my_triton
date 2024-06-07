import glob
import os
import subprocess
import sys
import tempfile

import numpy as np

import triton
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.driver import include_dir, library_dirs

tmp_dir = "/root/my_triton/AOT/2d9pt_tcstencil_layout"
dtype = "fp16"
kernel_path = "/root/my_triton/AOT/2d9pt_tcstencil_layout/2d9pt_tcstencil_layout.py"
sig = "*fp16, *fp16, 7200, 7200, *fp16, *fp16, 2, 16, 16"
kernel_name = "kernel"
grid = "202500, 1, 1" 
warp_num = 4


def _compile_kernel(dir, signature, kernel_name, out_name, out_path, num_warps, grid, kernel_path):
    compiler_path = os.path.join(triton.tools.__path__[0], "compile.py")

    subprocess.run(
        [
            sys.executable,
            compiler_path,
            "-n",
            kernel_name,
            "--signature",
            signature,
            "--out-name",
            out_name,
            "-o",
            out_path,
            "-w",
            str(num_warps),
            "-g",
            grid,
            kernel_path,
        ],
        check=True,
        cwd=dir,
    )


def compile_aot_kernels(dir, kernel_path, sig, name, grid, warp_num):

    # sig = "*fp16, *fp16, 520, 520, *fp16, *fp16, 2, 16, 16, 12, 12"
    # name = "kernel"
    # grid = "1849, 1, 1"
    _compile_kernel(
        dir=dir,
        signature=sig,
        kernel_name=name,
        out_name=name,
        out_path=name,
        num_warps=warp_num,
        grid=grid,
        kernel_path=kernel_path,
    )

def link_aot_kernels(dir):
    linker_path = os.path.join(triton.tools.__path__[0], "link.py")

    # link all desired configs
    h_files = glob.glob(os.path.join(dir, "*.h"))
    subprocess.run([sys.executable, linker_path] + h_files + ["-o", "kernel"], check=True, cwd=dir)

def gen_kernel_library(dir, libname):
    c_files = glob.glob(os.path.join(dir, "*.c"))
    subprocess.run(
        ["gcc"] + c_files + ["-I", include_dir[0], "-c", "-fPIC"],
        check=True,
        cwd=dir,
    )
    o_files = glob.glob(os.path.join(dir, "*.o"))

    command = ["gcc", *o_files, "-shared", "-o", libname]
    for lib_dir in library_dirs():
        command.extend(["-L", lib_dir])
    subprocess.run(command, check=True, cwd=dir)


def gen_test_bin(dir, exe="test", algo_id=0):

    command = ["gcc", "main.c"]
    for inc_dir in include_dir:
        command.extend(["-I", inc_dir])
    for lib_dir in library_dirs():
        command.extend(["-L", lib_dir])
    command.extend(["-l", "cuda", "-L", dir, "-l", "kernel", "-o", exe])
    subprocess.run(command, check=True, cwd=dir)

compile_aot_kernels(tmp_dir, kernel_path, sig, kernel_name,grid,warp_num)
link_aot_kernels(tmp_dir)

# compile test case
os.system("mv main.c main.cp")
gen_kernel_library(tmp_dir, "libkernel.so")
os.system("mv main.cp main.c")
gen_test_bin(tmp_dir)

# run test case
env = os.environ.copy()
env["LD_LIBRARY_PATH"] = tmp_dir
subprocess.run(["./test"], env=env, check=True, cwd=tmp_dir)

# # read data and compare against reference
# c = np.genfromtxt(c_path, delimiter=",", dtype=np.int32)
# c_tri = c.reshape((M, N)).view(np.float32)
# c_ref = np.matmul(a.astype(np.float32), b.astype(np.float32))
# np.testing.assert_allclose(c_tri, c_ref * c_ref, atol=1e-4, rtol=0.0)