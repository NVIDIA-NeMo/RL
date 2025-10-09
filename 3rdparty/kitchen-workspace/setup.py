import os
import re
import glob
import shutil
import pathlib
import subprocess
import setuptools
from typing import Tuple
import torch.utils.cpp_extension


def get_version() -> str:
    # read version from version.txt
    root_path = pathlib.Path(__file__).resolve().parent / "kitchen"
    with open(root_path / "version.txt", "r") as f:
        version = f.readline().strip()

    # add git version if not in release
    add_git_version = not bool(int(os.getenv("KITCHEN_NO_GIT_VERSION", "0")))
    if add_git_version:
        try:
            version_suffix = (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
                .decode("ascii")
                .strip()
            )
            version = f"{version}+{version_suffix}"
        except Exception:
            pass

    return version


def get_cuda_version() -> Tuple[int, ...]:
    # Try to locate NVCC
    nvcc_bin = None

    # Check in CUDA_HOME environment variable
    cuda_home = os.getenv("CUDA_HOME")
    if cuda_home:
        nvcc_bin = pathlib.Path(cuda_home) / "bin" / "nvcc"

    # Check in PATH if not found in CUDA_HOME
    if nvcc_bin is None or not nvcc_bin.is_file():
        nvcc_path = shutil.which("nvcc")
        if nvcc_path:
            nvcc_bin = pathlib.Path(nvcc_path)

    # Check in default directory
    if nvcc_bin is None or not nvcc_bin.is_file():
        nvcc_bin = pathlib.Path("/usr/local/cuda/bin/nvcc")

    if not nvcc_bin.is_file():
        raise FileNotFoundError(
            "Could not locate NVCC. Please ensure CUDA is installed and accessible."
        )

    # Run nvcc to get the version
    try:
        output = subprocess.run(
            [nvcc_bin, "-V"],
            capture_output=True,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to run NVCC: {e}")

    # Parse the version string
    match = re.search(r"release\s*([\d.]+)", output.stdout)
    if not match:
        raise ValueError("Could not parse CUDA version from NVCC output.")

    version = tuple(map(int, match.group(1).split(".")))
    return version


def assert_cuda_version(version: Tuple[int, ...]) -> None:
    """
    Validates that the provided CUDA version meets Kitchen's minimum requirements.
    Defaults to 12.6, or 12.4 if KITCHEN_ALLOW_CUDA_VERSION_12_4=1.
    """
    allow_cuda_version_12_4 = os.getenv("KITCHEN_ALLOW_CUDA_VERSION_12_4", "0") == "1"
    if allow_cuda_version_12_4:
        lowest_cuda_version = (12, 4)
    else:
        lowest_cuda_version = (12, 6)
    if version < lowest_cuda_version:
        raise RuntimeError(
            f"Kitchen requires CUDA {lowest_cuda_version} or newer. Got {version}"
        )


def get_cxx_flags(debug_mode: bool) -> list[str]:
    base_cxx_flags = [
        "-fvisibility=hidden",
        "-fdiagnostics-color=always",
        "-std=c++20",
    ]
    if debug_mode:
        return base_cxx_flags + ["-g", "-lineinfo", "-O0"]
    else:
        return base_cxx_flags + ["-O3"]


def get_nvcc_flags(debug_mode: bool, for_cutlass_gemm: bool) -> list[str]:
    base_nvcc_flags = [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-std=c++20",

        # suppress warning from libtorch headers
        # https://github.com/pytorch/pytorch/issues/98917
        "-diag-suppress=3189",
        "-diag-suppress=2908",  # suppress warning from cutlass sm100_blockscaled_mma_warpspecialized.hpp
    ]
    if debug_mode:
        if for_cutlass_gemm:
            # -G hangs the compilation process for CUTLASS due to template heavy
            # nature of the code.
            # CUTLASS is different since it has it's own way to debug such as
            # trace and compute-sanitizer etc.
            return base_nvcc_flags + ["-g", "-lineinfo", "-O0"]
        else:
            return base_nvcc_flags + ["-g", "-lineinfo", "-O0", "-G"]
    else:
        if for_cutlass_gemm:
            # TODO(Frank): Remove `-DNDEBUG when CUTLASS team fix the bug
            # https://nvidia.slack.com/archives/C8KNB5Y2E/p1747350354638559`
            # TODO(Frank): Does `-DNDEBUG` really just applied to .cu files?
            return base_nvcc_flags + ["-O3", "-DNDEBUG"]
        else:
            return base_nvcc_flags + ["-O3"]


def get_nvcc_gencode_flags(cuda_version: Tuple[int, ...]) -> list[str]:
    assert_cuda_version(cuda_version)
    # add nvcc flags for specific architectures
    if cuda_version >= (12, 8):
        cuda_archs = os.getenv("KITCHEN_CUDA_ARCHS", "90a;100a")
    else:
        cuda_archs = os.getenv("KITCHEN_CUDA_ARCHS", "90a")
    gencode_flags = []

    for arch in cuda_archs.split(";"):
        # TODO(Frank): Do we need `arch=compute_{arch},code=compute_{arch}`?
        gencode_flags.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
    return gencode_flags


def setup_extensions() -> setuptools.Extension:
    debug_mode = os.getenv("DEBUG", "0") == "1"

    # compiler flags
    cxx_flags = get_cxx_flags(debug_mode)
    ext_nvcc_flags = get_nvcc_flags(debug_mode, False)
    cutlass_gemm_ext_nvcc_flags = get_nvcc_flags(debug_mode, True)
    # Necessary linker flags. We need those for our extension regardless of
    # what PyTorch depends on.
    # extra_link_args = ["-lcuda", "-lcudart", "-lcublasLt"]

    # cublaslt_lib_path = "/usr/local/cuda/targets/x86_64-linux/lib/libcublasLt.so.12"
    # extra_link_args = [
    #     "-lcuda",
    #     "-lcudart",
    #     cublaslt_lib_path  # <-- The direct path to the file
    # ]

    cuda_lib_dir = "/usr/local/cuda/lib64"
    cublaslt_lib_path = f"{cuda_lib_dir}/libcublasLt.so.12.9.0.13"

    extra_link_args = [
        "-lcuda",
        "-lcudart",
        # Set the runtime path to its directory so it's found when you import
        f"-Wl,-rpath,{cuda_lib_dir}", f"-L{cuda_lib_dir}", "-lcublasLt", 
    ]


    # version-dependent CUDA options
    try:
        cuda_version = get_cuda_version()
    except FileNotFoundError:
        raise FileNotFoundError("Could not determine CUDA Toolkit version")
    else:
        gencode_flags = get_nvcc_gencode_flags(cuda_version)
        ext_nvcc_flags.extend(gencode_flags)
        cutlass_gemm_ext_nvcc_flags.extend(gencode_flags)

    # define sources and include directories
    root_dir = pathlib.Path(__file__).resolve().parent / "kitchen"
    extensions_dir = root_dir / "kitchen" / "csrc"

    src_dir = "kitchen/kitchen/csrc"
    cutlass_gemm_sources = set(
        glob.glob(
            os.path.join(src_dir, "ops/gemm/cutlass_gemm/*.cu"), recursive=True
        )
    ) | set(
        [
            os.path.join(
                src_dir, "ops/gemm/cutlass_gemm/cutlass_gemm_extensions.cpp"
            )
        ]
    )

    ext_sources = set(
        glob.glob(os.path.join(src_dir, "**/*.cpp"), recursive=True)
    )
    ext_sources |= set(
        glob.glob(os.path.join(src_dir, "**/*.cu"), recursive=True)
    )
    # Exclude cutlass_gemm files
    ext_sources = ext_sources - cutlass_gemm_sources

    ext_include_dirs = [
        '/usr/local/cuda/include',
        str(extensions_dir)
    ]

    cutlass_gemm_include_dirs = ext_include_dirs + [
        str(root_dir / "third_party" / "cutlass" / "include"),
        str(root_dir / "third_party" / "cutlass" / "tools" / "util" / "include"),
    ]

    # construct the extension
    print(f"DEBUG {ext_sources} ext_include_dirs")
    ext_module = torch.utils.cpp_extension.CUDAExtension(
        name="kitchen.ext",
        sources=ext_sources,
        include_dirs=ext_include_dirs,
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": ext_nvcc_flags,
        },
        extra_link_args=extra_link_args,
    )

    cutlass_gemm_ext_module = torch.utils.cpp_extension.CUDAExtension(
        name="kitchen.cutlass_gemm_ext",
        sources=cutlass_gemm_sources,
        include_dirs=cutlass_gemm_include_dirs,
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": cutlass_gemm_ext_nvcc_flags,
        },
        extra_link_args=extra_link_args,
    )
    return [ext_module, cutlass_gemm_ext_module]


setuptools.setup(
    name="nvidia-kitchen",
    version=get_version(),
    packages=["kitchen", "nvidia_kitchen"],
    package_dir={
        "kitchen": "kitchen/kitchen",
        "nvidia_kitchen": "kitchen/nvidia_kitchen",
    },
    package_data={
        "kitchen": ["ops/data/*.json"],  # Include all JSON files in ops/data

    },
    include_package_data=True,
    description="Nvidia Kitchen",
    ext_modules=setup_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    setup_requires=["cmake", "ninja"],
    # scipy is for hadamard transform
    install_requires=["torch", "scipy==1.15.2"],
    url="https://gitlab-master.nvidia.com/compute/chef/kitchen",
)
