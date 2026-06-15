#!/usr/bin/env bash
# Lists all bundled codecs in the current container environment.
# For each library also shows: install path, and which package pulled it in.
# Run inside the container: bash tools/list_codecs.sh

set -euo pipefail

section() {
    echo ""
    echo "========================================"
    echo "  $1"
    echo "========================================"
}

# Print install path + reverse-dependency info for a Python package.
# Usage: py_provenance <import_name> <pip_package_name>
py_provenance() {
    local import_name="$1"
    local pip_name="$2"
    python3 - <<PYEOF
import importlib, subprocess, sys

# File path
try:
    mod = importlib.import_module("$import_name")
    path = getattr(mod, "__file__", None) or getattr(mod, "__path__", ["(namespace)"])[0]
    print(f"  path:        {path}")
except ImportError:
    print("  (not importable)")
    sys.exit(0)

# pip show: Location + Required-by
try:
    out = subprocess.check_output(
        [sys.executable, "-m", "pip", "show", "$pip_name"],
        stderr=subprocess.DEVNULL, text=True
    )
    for line in out.splitlines():
        if line.startswith("Location:") or line.startswith("Required-by:"):
            print(f"  {line}")
except Exception:
    pass

# pipdeptree reverse tree (if available)
try:
    tree = subprocess.check_output(
        [sys.executable, "-m", "pipdeptree", "--reverse", "--packages", "$pip_name"],
        stderr=subprocess.DEVNULL, text=True
    ).strip()
    if tree:
        print("  reverse-dep tree:")
        for l in tree.splitlines():
            print(f"    {l}")
except Exception:
    pass
PYEOF
}

# ---------------------------------------------------------------------------
# 1. Video / Audio codecs (ffmpeg)
# ---------------------------------------------------------------------------
section "VIDEO/AUDIO CODECS (ffmpeg)"
if command -v ffmpeg &>/dev/null; then
    echo "  path: $(command -v ffmpeg)"
    echo ""
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
    echo "  path: $(command -v gst-inspect-1.0)"
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
import sys, importlib
for mod in ["zlib", "bz2", "lzma", "gzip", "zipfile"]:
    try:
        m = importlib.import_module(mod)
        path = getattr(m, "__file__", "(built-in)")
        print(f"  {mod}: {path}")
    except ImportError:
        print(f"  {mod}: not available")
EOF

# ---------------------------------------------------------------------------
# 5. Compression codecs — third-party Python
# ---------------------------------------------------------------------------
section "COMPRESSION CODECS (third-party Python)"

echo "[blosc]"
python3 - <<'EOF'
try:
    import blosc
    print("  codecs:", blosc.compressor_list())
except ImportError as e:
    print(f"  not available ({e})")
EOF
py_provenance blosc blosc

echo ""
echo "[numcodecs]"
python3 - <<'EOF'
try:
    import numcodecs
    print("  codecs:", numcodecs.available_codecs())
except ImportError as e:
    print(f"  not available ({e})")
EOF
py_provenance numcodecs numcodecs

echo ""
echo "[pyarrow]"
python3 - <<'EOF'
try:
    import pyarrow
    supported = [c for c in ['lz4','zstd','snappy','brotli','gzip','bz2'] if pyarrow.Codec.is_available(c)]
    print("  codecs:", supported)
except ImportError as e:
    print(f"  not available ({e})")
EOF
py_provenance pyarrow pyarrow

echo ""
echo "[zarr]"
python3 - <<'EOF'
try:
    import zarr
    reg = list(zarr.codec_registry.keys()) if hasattr(zarr, 'codec_registry') else 'N/A'
    print("  codec registry:", reg)
except ImportError as e:
    print(f"  not available ({e})")
EOF
py_provenance zarr zarr

echo ""
echo "[h5py]"
python3 - <<'EOF'
try:
    import h5py
    print(f"  h5py {h5py.__version__} | HDF5 {h5py.version.hdf5_version}")
except ImportError as e:
    print(f"  not available ({e})")
EOF
py_provenance h5py h5py

# ---------------------------------------------------------------------------
# 6. Image codecs — Pillow
# ---------------------------------------------------------------------------
section "IMAGE CODECS (Pillow)"
python3 - <<'EOF'
try:
    from PIL import features
    features.pilinfo()
except ImportError as e:
    print(f"not available ({e})")
EOF
py_provenance PIL Pillow

# ---------------------------------------------------------------------------
# 7. Image codecs — OpenCV
# ---------------------------------------------------------------------------
section "IMAGE CODECS (OpenCV)"
python3 - <<'EOF'
try:
    import cv2
    print(cv2.getBuildInformation())
except ImportError as e:
    print(f"not available ({e})")
EOF
py_provenance cv2 opencv-python

# ---------------------------------------------------------------------------
# 8. Image codecs — imagecodecs
# ---------------------------------------------------------------------------
section "IMAGE CODECS (imagecodecs)"
python3 - <<'EOF'
try:
    import imagecodecs
    print(imagecodecs.available())
except ImportError as e:
    print(f"not available ({e})")
EOF
py_provenance imagecodecs imagecodecs

# ---------------------------------------------------------------------------
# 9. Image codecs — imageio
# ---------------------------------------------------------------------------
section "IMAGE CODECS (imageio)"
python3 - <<'EOF'
try:
    import imageio
    print(imageio.formats.show())
except ImportError as e:
    print(f"not available ({e})")
EOF
py_provenance imageio imageio

# ---------------------------------------------------------------------------
# 10. Video codecs — PyAV
# ---------------------------------------------------------------------------
section "VIDEO CODECS (PyAV)"
python3 - <<'EOF'
try:
    import av
    print("available codecs:", sorted(av.codec.codecs_available))
except ImportError as e:
    print(f"not available ({e})")
EOF
py_provenance av av

# ---------------------------------------------------------------------------
# 11. Video codecs — decord
# ---------------------------------------------------------------------------
section "VIDEO CODECS (decord)"
python3 - <<'EOF'
try:
    import decord
    print("version:", decord.__version__)
except ImportError as e:
    print(f"not available ({e})")
EOF
py_provenance decord decord

# ---------------------------------------------------------------------------
# 12. TIFF codecs — tifffile
# ---------------------------------------------------------------------------
section "IMAGE CODECS (tifffile)"
python3 - <<'EOF'
try:
    import tifffile
    print("TIFF compressions:", list(tifffile.TIFF.COMPRESSION.__members__.keys()))
except ImportError as e:
    print(f"not available ({e})")
EOF
py_provenance tifffile tifffile

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
