#!/usr/bin/env bash
# Lists all bundled codecs in the current container environment.
# Run inside the container: bash tools/list_codecs.sh

set -euo pipefail

section() {
    echo ""
    echo "========================================"
    echo "  $1"
    echo "========================================"
}

# ---------------------------------------------------------------------------
# 1. Video / Audio codecs (ffmpeg)
# ---------------------------------------------------------------------------
section "VIDEO/AUDIO CODECS (ffmpeg)"
if command -v ffmpeg &>/dev/null; then
    echo "--- All codecs ---"
    ffmpeg -codecs 2>/dev/null || true
    echo ""
    echo "--- Encoders ---"
    ffmpeg -encoders 2>/dev/null || true
    echo ""
    echo "--- Decoders ---"
    ffmpeg -decoders 2>/dev/null || true
    echo ""
    echo "--- Container formats ---"
    ffmpeg -formats 2>/dev/null || true
else
    echo "ffmpeg not found — skipping"
fi

# ---------------------------------------------------------------------------
# 2. GStreamer
# ---------------------------------------------------------------------------
section "VIDEO/AUDIO CODECS (GStreamer)"
if command -v gst-inspect-1.0 &>/dev/null; then
    gst-inspect-1.0 --print-all 2>/dev/null | grep -i codec || true
else
    echo "gst-inspect-1.0 not found — skipping"
fi

# ---------------------------------------------------------------------------
# 3. Compression codecs — system libraries
# ---------------------------------------------------------------------------
section "COMPRESSION CODECS (system libraries)"
if command -v ldconfig &>/dev/null; then
    ldconfig -p 2>/dev/null | grep -E 'lz4|zstd|snappy|brotli|lzma|zlib|bz2' || echo "none found"
else
    echo "ldconfig not available"
fi

# ---------------------------------------------------------------------------
# 4. Compression codecs — Python stdlib
# ---------------------------------------------------------------------------
section "COMPRESSION CODECS (Python stdlib)"
python3 - <<'EOF'
import sys
codecs = []
for mod in ["zlib", "bz2", "lzma", "gzip", "zipfile"]:
    try:
        __import__(mod)
        codecs.append(mod)
    except ImportError:
        pass
print("Available:", ", ".join(codecs))
EOF

# ---------------------------------------------------------------------------
# 5. Compression codecs — third-party Python
# ---------------------------------------------------------------------------
section "COMPRESSION CODECS (third-party Python)"
python3 - <<'EOF'
checks = {
    "blosc":     "import blosc; print('blosc compressors:', blosc.compressor_list())",
    "numcodecs": "import numcodecs; print('numcodecs:', numcodecs.available_codecs())",
    "pyarrow":   "import pyarrow; codecs=['lz4','zstd','snappy','brotli','gzip','bz2']; print('pyarrow available:', [c for c in codecs if pyarrow.Codec.is_available(c)])",
    "zarr":      "import zarr; print('zarr codec registry:', list(zarr.codec_registry.keys()) if hasattr(zarr,'codec_registry') else 'N/A')",
    "h5py":      "import h5py; print('h5py version:', h5py.__version__, '| HDF5:', h5py.version.hdf5_version)",
}
for lib, code in checks.items():
    try:
        exec(code)
    except Exception as e:
        print(f"{lib}: not available ({e})")
EOF

# ---------------------------------------------------------------------------
# 6. Image codecs — Pillow
# ---------------------------------------------------------------------------
section "IMAGE CODECS (Pillow)"
python3 - <<'EOF'
try:
    from PIL import features
    features.pilinfo()
except ImportError:
    print("Pillow not installed")
EOF

# ---------------------------------------------------------------------------
# 7. Image codecs — OpenCV
# ---------------------------------------------------------------------------
section "IMAGE CODECS (OpenCV)"
python3 - <<'EOF'
try:
    import cv2
    print(cv2.getBuildInformation())
except ImportError:
    print("opencv-python not installed")
EOF

# ---------------------------------------------------------------------------
# 8. Image codecs — imagecodecs
# ---------------------------------------------------------------------------
section "IMAGE CODECS (imagecodecs)"
python3 - <<'EOF'
try:
    import imagecodecs
    print(imagecodecs.available())
except ImportError:
    print("imagecodecs not installed")
EOF

# ---------------------------------------------------------------------------
# 9. Image codecs — imageio
# ---------------------------------------------------------------------------
section "IMAGE CODECS (imageio)"
python3 - <<'EOF'
try:
    import imageio
    print(imageio.formats.show())
except ImportError:
    print("imageio not installed")
EOF

# ---------------------------------------------------------------------------
# 10. Video codecs — PyAV
# ---------------------------------------------------------------------------
section "VIDEO CODECS (PyAV)"
python3 - <<'EOF'
try:
    import av
    print("PyAV available codecs:", sorted(av.codec.codecs_available))
except ImportError:
    print("PyAV not installed")
EOF

# ---------------------------------------------------------------------------
# 11. Video codecs — decord
# ---------------------------------------------------------------------------
section "VIDEO CODECS (decord)"
python3 - <<'EOF'
try:
    import decord
    print("decord version:", decord.__version__)
except ImportError:
    print("decord not installed")
EOF

# ---------------------------------------------------------------------------
# 12. TIFF codecs — tifffile
# ---------------------------------------------------------------------------
section "IMAGE CODECS (tifffile / TIFF compressions)"
python3 - <<'EOF'
try:
    import tifffile
    print("TIFF compressions:", list(tifffile.TIFF.COMPRESSION.__members__.keys()))
except ImportError:
    print("tifffile not installed")
EOF

# ---------------------------------------------------------------------------
# 13. System packages (apt/dpkg)
# ---------------------------------------------------------------------------
section "SYSTEM PACKAGES (codec-related)"
if command -v dpkg &>/dev/null; then
    dpkg -l 2>/dev/null | grep -E 'ffmpeg|codec|libav|x264|x265|vpx|aom|opus|vorbis|theora|libde265|hevc' || echo "none found"
else
    echo "dpkg not available"
fi

echo ""
echo "Done."
