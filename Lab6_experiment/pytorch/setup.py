from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parents[1]


def extension(name, cpp, cu):
    return CUDAExtension(
        name,
        [
            str(ROOT / "cpp" / cpp),
            str(ROOT / "cuda" / cu),
        ],
        include_dirs=[str(ROOT / "include")],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", "--expt-relaxed-constexpr"],
        },
    )


setup(
    name="lab6_custom_conv2d",
    packages=find_packages(),
    ext_modules=[
        extension(
            "conv2d_baseline_fp32",
            "conv2d_baseline_fp32.cpp",
            "conv2d_baseline_kernel_fp32.cu",
        ),
        extension(
            "conv2d_optim_fp32",
            "conv2d_optim_fp32.cpp",
            "conv2d_optim_kernel_fp32.cu",
        ),
        extension(
            "conv2d_optim_fp16",
            "conv2d_optim_fp16.cpp",
            "conv2d_optim_kernel_fp16.cu",
        ),
        extension(
            "conv2d_optim_wmma",
            "conv2d_optim_wmma.cpp",
            "conv2d_optim_kernel_wmma.cu",
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
