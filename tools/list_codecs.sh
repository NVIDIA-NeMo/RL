#!/usr/bin/env bash
# Lists all bundled codecs in the current container environment.
# For each library: install path, pip provenance, reverse-dep tree.
# Scans auditwheel *.libs/ dirs for bundled .so files (e.g. libx264 inside PyAV).
# Ends with a ROYALTY-BEARING CODEC SUMMARY for OSRB point 12.
# Run inside the container: bash tools/list_codecs.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve python and ffmpeg — PATH is often minimal inside srun containers
# ---------------------------------------------------------------------------
PYTHON=""
for candidate in /opt/nemo_rl_venv/bin/python python3 python /usr/bin/python3 /usr/local/bin/python3 /opt/conda/bin/python3 /opt/conda/bin/python; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done
if [[ -z "$PYTHON" ]]; then
    echo "ERROR: no Python interpreter found. Tried: /opt/nemo_rl_venv/bin/python python3 python /usr/bin/python3 /usr/local/bin/python3 /opt/conda/bin/python3"
    exit 1
fi
echo "Using Python: $PYTHON ($(command -v "$PYTHON"))"

FFMPEG=""
for candidate in ffmpeg /usr/bin/ffmpeg /usr/local/bin/ffmpeg /opt/conda/bin/ffmpeg; do
    if command -v "$candidate" &>/dev/null; then
        FFMPEG="$candidate"
        break
    fi
done
[[ -z "$FFMPEG" ]] && echo "ffmpeg not found on PATH or common locations — video codec sections will be skipped"

# Royalty-bearing codec pattern (AAC, H.264, H.265 and all variants)
ROYALTY_PATTERN='libx264|libx265|libh264|libh265|libaac|libfaac|libfdk.aac|libopenh264|libde265|hevc|x264|x265|avc.*enc|h264.*enc|h265.*enc|hvec'

# Accumulate royalty hits across all sections into this file
ROYALTY_TMP=$(mktemp)
trap 'rm -f "$ROYALTY_TMP"' EXIT

flag_royalty() {
    # flag_royalty <section_name> <text>
    local section="$1"
    local text="$2"
    if echo "$text" | grep -iE "$ROYALTY_PATTERN" &>/dev/null; then
        echo "$text" | grep -iE "$ROYALTY_PATTERN" | while IFS= read -r line; do
            echo "  [$section] $line" >> "$ROYALTY_TMP"
        done
    fi
}

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
    "$PYTHON" - <<PYEOF
import importlib, subprocess, sys

try:
    mod = importlib.import_module("$import_name")
    path = getattr(mod, "__file__", None) or getattr(mod, "__path__", ["(namespace)"])[0]
    print(f"  path:        {path}")
except ImportError:
    print("  (not importable)")
    sys.exit(0)

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
if [[ -n "$FFMPEG" ]]; then
    echo "  path: $FFMPEG"
    echo ""
    echo "--- Build configuration (configure flags) ---"
    "$FFMPEG" -buildconf 2>/dev/null || true
    echo ""
    echo "--- All codecs ---"
    FFMPEG_CODECS=$("$FFMPEG" -codecs 2>/dev/null || true)
    echo "$FFMPEG_CODECS"
    flag_royalty "ffmpeg -codecs" "$FFMPEG_CODECS"
    echo ""
    echo "--- Encoders ---"
    "$FFMPEG" -encoders 2>/dev/null || true
    echo ""
    echo "--- Decoders ---"
    "$FFMPEG" -decoders 2>/dev/null || true
    echo ""
    echo "--- Container formats ---"
    "$FFMPEG" -formats 2>/dev/null || true
else
    echo "ffmpeg not found — skipping"
fi

# ---------------------------------------------------------------------------
# 2. GStreamer
# ---------------------------------------------------------------------------
section "VIDEO/AUDIO CODECS (GStreamer)"
if command -v gst-inspect-1.0 &>/dev/null; then
    echo "  path: $(command -v gst-inspect-1.0)"
    GST_OUT=$(gst-inspect-1.0 --print-all 2>/dev/null | grep -i codec || true)
    echo "$GST_OUT"
    flag_royalty "GStreamer" "$GST_OUT"
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
"$PYTHON" - <<'EOF'
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
"$PYTHON" - <<'EOF'
try:
    import blosc
    print("  codecs:", blosc.compressor_list())
except ImportError as e:
    print(f"  not available ({e})")
EOF
py_provenance blosc blosc

echo ""
echo "[numcodecs]"
"$PYTHON" - <<'EOF'
try:
    import numcodecs
    print("  codecs:", numcodecs.available_codecs())
except ImportError as e:
    print(f"  not available ({e})")
EOF
py_provenance numcodecs numcodecs

echo ""
echo "[pyarrow]"
"$PYTHON" - <<'EOF'
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
"$PYTHON" - <<'EOF'
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
"$PYTHON" - <<'EOF'
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
"$PYTHON" - <<'EOF'
try:
    from PIL import features
    features.pilinfo()
except ImportError as e:
    print(f"not available ({e})")
EOF
py_provenance PIL Pillow

# ---------------------------------------------------------------------------
# 7. Image codecs — OpenCV (highlights codec-relevant lines)
# ---------------------------------------------------------------------------
section "IMAGE CODECS (OpenCV)"
CV2_OUT=$("$PYTHON" - <<'EOF'
try:
    import cv2
    print(cv2.getBuildInformation())
except ImportError as e:
    print(f"not available ({e})")
EOF
)
echo "$CV2_OUT"
flag_royalty "OpenCV getBuildInformation" "$CV2_OUT"
py_provenance cv2 opencv-python

# ---------------------------------------------------------------------------
# 8. Image codecs — imagecodecs
# ---------------------------------------------------------------------------
section "IMAGE CODECS (imagecodecs)"
"$PYTHON" - <<'EOF'
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
"$PYTHON" - <<'EOF'
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
PYAV_OUT=$("$PYTHON" - <<'EOF'
try:
    import av
    print("available codecs:", sorted(av.codec.codecs_available))
except ImportError as e:
    print(f"not available ({e})")
EOF
)
echo "$PYAV_OUT"
flag_royalty "PyAV codecs" "$PYAV_OUT"
py_provenance av av

# ---------------------------------------------------------------------------
# 11. Video codecs — decord
# ---------------------------------------------------------------------------
section "VIDEO CODECS (decord)"
"$PYTHON" - <<'EOF'
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
"$PYTHON" - <<'EOF'
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
    DPKG_OUT=$(dpkg -l 2>/dev/null | grep -E 'ffmpeg|codec|libav|x264|x265|vpx|aom|opus|vorbis|theora|libde265|hevc' || true)
    if [[ -n "$DPKG_OUT" ]]; then
        echo "$DPKG_OUT"
        flag_royalty "dpkg" "$DPKG_OUT"
    else
        echo "none found"
    fi
else
    echo "dpkg not available"
fi

# ---------------------------------------------------------------------------
# 14. Bundled .so libs inside Python wheel *.libs/ dirs (auditwheel packages)
#     e.g. av.libs/libx264-*.so, opencv_python.libs/libavcodec-*.so
# ---------------------------------------------------------------------------
section "BUNDLED .SO LIBS IN PYTHON WHEEL PACKAGES (auditwheel)"
"$PYTHON" - <<'PYEOF'
import sys, os, re, subprocess

site_dirs = [p for p in sys.path if "site-packages" in p and os.path.isdir(p)]
found_any = False

for site in site_dirs:
    for entry in sorted(os.listdir(site)):
        libs_dir = os.path.join(site, entry, "libs") if "." not in entry else os.path.join(site, entry + ".libs")
        # also handle e.g. av.libs as a direct dir under site-packages
        alt_libs_dir = os.path.join(site, entry.replace("-", "_") + ".libs")
        for d in [libs_dir, alt_libs_dir]:
            if not os.path.isdir(d):
                continue
            sos = sorted(f for f in os.listdir(d) if f.endswith(".so") or ".so." in f)
            if not sos:
                continue
            found_any = True
            pkg = entry
            print(f"\n  Package : {pkg}")
            print(f"  libs dir: {d}")
            for so in sos:
                print(f"    {so}")

if not found_any:
    print("  No *.libs/ directories found under site-packages.")
PYEOF

# Flag royalty-bearing .so names specifically
PYEOF_OUT=$("$PYTHON" - <<'PYEOF'
import sys, os

site_dirs = [p for p in sys.path if "site-packages" in p and os.path.isdir(p)]
import re
pattern = re.compile(r'libx264|libx265|libh264|libh265|libaac|libfaac|libfdk|libopenh264|libde265|libavcodec|libavformat|libavutil|libswscale', re.IGNORECASE)

for site in site_dirs:
    for entry in sorted(os.listdir(site)):
        for suffix in [".libs", "/libs"]:
            d = os.path.join(site, entry.replace("-","_") + ".libs") if suffix == ".libs" else os.path.join(site, entry, "libs")
            if not os.path.isdir(d):
                continue
            for so in os.listdir(d):
                if pattern.search(so):
                    print(f"  [{entry}] {d}/{so}")
PYEOF
)
if [[ -n "$PYEOF_OUT" ]]; then
    echo ""
    echo "  *** ROYALTY-RELEVANT .so files found inside wheel packages: ***"
    echo "$PYEOF_OUT"
    flag_royalty "auditwheel .so" "$PYEOF_OUT"
fi

# ---------------------------------------------------------------------------
# 15. ROYALTY-BEARING CODEC SUMMARY (for OSRB point 12)
# ---------------------------------------------------------------------------
section "ROYALTY-BEARING CODEC SUMMARY (OSRB point 12)"
echo "  Codecs matching royalty pattern: AAC, H.264/x264, H.265/x265/HEVC and variants"
echo ""
if [[ -s "$ROYALTY_TMP" ]]; then
    echo "  *** ACTION REQUIRED — the following royalty-bearing codecs were detected: ***"
    echo ""
    sort -u "$ROYALTY_TMP"
    echo ""
    echo "  Per NVIDIA OSRB policy:"
    echo "    - Remove unused codecs, OR"
    echo "    - Replace with HW codecs (NVENC/NVDEC), OR"
    echo "    - Obtain org3 management approval before distribution"
    echo "    - Disclose all findings in OSRB submission point 12"
else
    echo "  No royalty-bearing codecs detected."
    echo "  (Still verify manually — this script may not catch statically linked codecs.)"
fi

echo ""
echo "Done."
