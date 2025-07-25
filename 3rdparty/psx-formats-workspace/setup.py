import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

path = os.path.dirname(os.path.realpath(__file__)) + "/psx-formats"

gencodes = [
    "-gencode",
    "arch=compute_90,code=sm_90",
    "-gencode",
    "arch=compute_90,code=compute_90",
]

extra_warnings = [
    "-Xcompiler", "-Wall",
    "-Xcompiler", "-Wextra",
]

ptx_args = [
    "-Xptxas", "-v",
]

extra_compiler_args = [
    "--expt-relaxed-constexpr",
]

src_base_path = "src/"
src_file_list = [
    "extension.cu",
    "reduce_amax.cu",
    "convert.cu",
]
src_list = [src_base_path + f for f in src_file_list]

utils_base_path = "src/utils/"
utils_file_list = [
    "fused_bprop_clamp.cu",
    "uint8_to_fp8.cu",
    "utils_extension.cu",
    "clippy_1d.cu",
    "clippy_2d.cu",
]
utils_src_list = [utils_base_path + f for f in utils_file_list]
defines = []

include_dirs = [os.path.join(path, "src"), os.path.join(path, "external/psx_encodings/include")]

setup(
    name="psx_formats",
    version="0.2dev",
    package_dir={
        "psx_formats": "psx-formats/psx_formats",
    },
    ext_modules=[
        CUDAExtension(
            name="psx_formats._C",
            sources=[os.path.join("psx-formats", f) for f in src_list],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", *defines],
                "nvcc": ["-O3", "-std=c++17", *extra_warnings, *extra_compiler_args, *gencodes, *defines],
            },
            include_dirs=include_dirs,
        ),
        CUDAExtension(
            name="psx_formats.utils._C",
            sources=[os.path.join("psx-formats", f) for f in utils_src_list],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", *defines],
                "nvcc": ["-O3", "-std=c++17", *extra_warnings, *extra_compiler_args, *gencodes, *ptx_args, *defines],
            },
            include_dirs=include_dirs,
        ),
    ],
    setup_requires=["pytest-runner"],
    packages=["psx_formats"],
    cmdclass={"build_ext": BuildExtension},
)
