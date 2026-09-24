set -eou pipefail

# Note: # Note: TensorRT-LLM must be accessible at the same path on all nodes.
TRTLLM_SRC="${TRTLLM_SRC:-/workspace/TensorRT-LLM}"
WHEEL_OUTPUT_DIR=/tmp/trtllm-wheels
# SM arch list. Shares the BUILD_CUSTOM_TRTLLM_ARCH knob with build-custom-trtllm.sh.
# Default targets Blackwell (sm_100) and Blackwell-Ultra (sm_103)
# -- MUST stay in sync with build-custom-trtllm.sh / _backend.py's _DEFAULT_ARCH.
ARCH="${BUILD_CUSTOM_TRTLLM_ARCH:-100-real;103-real}"

mkdir -p "${WHEEL_OUTPUT_DIR}"

cd "${TRTLLM_SRC}"
echo "Commit: $(git rev-parse HEAD)"

# Run build_setup.sh with BASE_DIR pointing at the /tmp copy
# (registers safe.directory entries, installs git-lfs, pulls LFS objects)
# BASE_DIR="${TRTLLM_DIR}" bash /lustre/fsw/coreai_comparch_trtllm/erinh/build_setup.sh

# Same patches as build-custom-trtllm.sh (kept in sync by hand -- this script
# has no assert_patch_target guard, so a silent no-op here won't fail loudly).
sed -i 's|^setuptools<80$|setuptools|' requirements.txt

# Drop PyNvVideoCodec: no aarch64/py3.13 wheel exists in the pinned ~=2.1.0
# range, so build_wheel.py's pip install aborts before cmake ever runs.
sed -i '/^PyNvVideoCodec/d' requirements.txt

sed -i 's|COMMAND ${Python3_EXECUTABLE} setup_library.py develop --user|COMMAND bash -c "cp -f setup_library.py setup.py \&\& ${Python3_EXECUTABLE} setup_library.py develop"|' \
    cpp/tensorrt_llm/kernels/cutlass_kernels/CMakeLists.txt

# Fix nvshmem: it doesn't accept the 'f' suffix CMake >= 3.31 generates for
# Blackwell ('100f-real'). Substitute bare archs for the nvshmem cmake call
# only; DeepEP kernels keep the full arch string for FP4 support. Must match
# the ARCH default above (bare, semicolon-separated).
sed -i 's|-DCMAKE_CUDA_ARCHITECTURES:STRING=${DEEP_EP_CUDA_ARCHITECTURES}|-DCMAKE_CUDA_ARCHITECTURES:STRING=100\;103|' \
    cpp/tensorrt_llm/deep_ep/CMakeLists.txt

# NIXL is what cache_transceiver_backend=DEFAULT resolves to with UCX enabled
# (see build-custom-trtllm.sh). Fail here with a clear message rather than
# deep inside cmake's find_package(NIXL) if it's missing.
NIXL_ROOT_DIR="${NIXL_ROOT_DIR:-/opt/nvidia/nvda_nixl}"
if [[ ! -f "${NIXL_ROOT_DIR}/include/nixl.h" ]]; then
    echo "[ERROR] NIXL not found at ${NIXL_ROOT_DIR}. Install it (see the" \
         "NIXL_VERSION block in docker/Dockerfile), or point NIXL_ROOT_DIR at" \
         "an existing install." >&2
    exit 1
fi

echo "[INFO] Starting build: $(date)"
python3 scripts/build_wheel.py \
    -a "$ARCH" \
    -G Ninja \
    --clean \
    --nvrtc_dynamic_linking \
    -D "ENABLE_UCX=ON" \
    -D "BUILD_TESTS=OFF" \
    --nixl_root "${NIXL_ROOT_DIR}" \
    --dist_dir "${WHEEL_OUTPUT_DIR}"

echo "[INFO] Done: $(date)"
ls -lh "${WHEEL_OUTPUT_DIR}"/tensorrt_llm-*.whl
echo "Wheel is at ${WHEEL_OUTPUT_DIR} ~@~T copy to Lustre manually after freeing inodes"