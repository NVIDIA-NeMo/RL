set -eou pipefail

# Note: # Note: TensorRT-LLM must be accessible at the same path on all nodes.
TRTLLM_SRC="${TRTLLM_SRC:-/workspace/TensorRT-LLM}"
WHEEL_OUTPUT_DIR=/tmp/trtllm-wheels
# SM arch list. Shares the BUILD_CUSTOM_TRTLLM_ARCH knob with build-custom-trtllm.sh.
# Default targets GB200 (sm_100) + GB300 (sm_103).
ARCH="${BUILD_CUSTOM_TRTLLM_ARCH:-100-real;103-real}"

mkdir -p "${WHEEL_OUTPUT_DIR}"

cd "${TRTLLM_SRC}"
echo "Commit: $(git rev-parse HEAD)"

# Run build_setup.sh with BASE_DIR pointing at the /tmp copy
# (registers safe.directory entries, installs git-lfs, pulls LFS objects)
# BASE_DIR="${TRTLLM_DIR}" bash /lustre/fsw/coreai_comparch_trtllm/erinh/build_setup.sh

# Same patches as build-custom-trtllm.sh
sed -i 's|nvidia-modelopt\[torch\]~=0\.37\.0|nvidia-modelopt[torch]>=0.44.0a0|' requirements.txt
sed -i 's|^setuptools<80$|setuptools|' requirements.txt
sed -i 's|COMMAND ${Python3_EXECUTABLE} setup_library.py develop --user|COMMAND bash -c "cp -f setup_library.py setup.py \&\& ${Python3_EXECUTABLE} setup_library.py develop"|' \
    cpp/tensorrt_llm/kernels/cutlass_kernels/CMakeLists.txt

# Fix nvshmem: it doesn't accept '100f-real' (CMake >= 3.31 generates this for GB300).
# Hardcode '100' only for the nvshmem cmake call; DeepEP kernels keep '100f' for FP4 support.
sed -i 's|-DCMAKE_CUDA_ARCHITECTURES:STRING=${DEEP_EP_CUDA_ARCHITECTURES}|-DCMAKE_CUDA_ARCHITECTURES:STRING=100|' \
    cpp/tensorrt_llm/deep_ep/CMakeLists.txt

echo "[INFO] Starting build: $(date)"
python3 scripts/build_wheel.py \
    -a "$ARCH" \
    -G Ninja \
    --clean \
    --nvrtc_dynamic_linking \
    -D "ENABLE_UCX=OFF" \
    --dist_dir "${WHEEL_OUTPUT_DIR}"

echo "[INFO] Done: $(date)"
ls -lh "${WHEEL_OUTPUT_DIR}"/tensorrt_llm-*.whl
echo "Wheel is at ${WHEEL_OUTPUT_DIR} ~@~T copy to Lustre manually after freeing inodes"