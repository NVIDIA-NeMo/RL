"""Build the deployment wheel using only the Python standard library.

The external-service login node may not have access to PyPI. This builder keeps
the launch path offline while ``pyproject.toml`` remains available for normal
developer builds.
"""

import base64
import csv
import hashlib
import io
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
PACKAGE_ROOT = PROJECT_ROOT / "src/nemo_rl_vllm_prefix_plugin"
DISTRIBUTION = "nemo_rl_vllm_prefix_plugin"
VERSION = "0.2.0"
DIST_INFO = f"{DISTRIBUTION}-{VERSION}.dist-info"
WHEEL_NAME = f"{DISTRIBUTION}-{VERSION}-py3-none-any.whl"


def _record_digest(contents: bytes) -> str:
    digest = hashlib.sha256(contents).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return f"sha256={encoded}"


def build_wheel(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = output_dir / WHEEL_NAME
    files: dict[str, bytes] = {}

    for source_path in sorted(PACKAGE_ROOT.glob("*.py")):
        archive_path = f"{DISTRIBUTION}/{source_path.name}"
        files[archive_path] = source_path.read_bytes()

    files[f"{DIST_INFO}/METADATA"] = (
        "Metadata-Version: 2.3\n"
        "Name: nemo-rl-vllm-prefix-plugin\n"
        f"Version: {VERSION}\n"
        "Summary: NeMo RL prefix-token and external-staging support for vLLM\n"
        "Requires-Python: >=3.10\n"
        "Requires-Dist: vllm==0.29.0\n"
        "\n"
    ).encode()
    files[f"{DIST_INFO}/WHEEL"] = (
        "Wheel-Version: 1.0\n"
        "Generator: nemo-rl-vllm-prefix-plugin\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n"
        "\n"
    ).encode()
    files[f"{DIST_INFO}/entry_points.txt"] = (
        "[vllm.endpoint_plugins]\n"
        "nemo_rl_prefix_api = "
        "nemo_rl_vllm_prefix_plugin.plugin:NeMoRLPrefixEndpointPlugin\n"
    ).encode()

    record_buffer = io.StringIO(newline="")
    record_writer = csv.writer(record_buffer, lineterminator="\n")
    for archive_path, contents in files.items():
        record_writer.writerow(
            [archive_path, _record_digest(contents), str(len(contents))]
        )
    record_path = f"{DIST_INFO}/RECORD"
    record_writer.writerow([record_path, "", ""])
    files[record_path] = record_buffer.getvalue().encode()

    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for archive_path, contents in files.items():
            wheel.writestr(archive_path, contents)

    return wheel_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} OUTPUT_DIR")
    print(build_wheel(Path(sys.argv[1]).resolve()))
