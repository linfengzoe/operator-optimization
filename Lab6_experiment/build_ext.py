# -*- coding: utf-8 -*-
"""
JIT 编译三个算子，绕过中文路径导致的 ninja 问题
"""
import os, sys, shutil
from torch.utils.cpp_extension import load

base = os.path.dirname(os.path.abspath(__file__))

def build(name, cpp, cu, extra_cflags=None):
    print(f"\n{'='*50}")
    print(f"编译 {name} ...")
    extra = extra_cflags or []
    mod = load(
        name=name,
        sources=[os.path.join(base, cpp), os.path.join(base, cu)],
        extra_include_paths=[os.path.join(base, 'include')],
        extra_cflags=extra,
        extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr'],
        verbose=True,
        build_directory=os.path.join(base, f'build_jit_{name}'),
    )
    print(f"✓ {name} 编译成功")
    return mod

if __name__ == '__main__':
    baseline = build('conv2d_baseline_fp32',
                     'cpp/conv2d_baseline_fp32.cpp',
                     'cuda/conv2d_baseline_kernel_fp32.cu')

    fp32 = build('conv2d_optim_fp32',
                 'cpp/conv2d_optim_fp32.cpp',
                 'cuda/conv2d_optim_kernel_fp32.cu')

    fp16 = build('conv2d_optim_fp16',
                 'cpp/conv2d_optim_fp16.cpp',
                 'cuda/conv2d_optim_kernel_fp16.cu')

    wmma = build('conv2d_optim_wmma',
                 'cpp/conv2d_optim_wmma.cpp',
                 'cuda/conv2d_optim_kernel_wmma.cu')

    print("\n所有算子编译完成！")
