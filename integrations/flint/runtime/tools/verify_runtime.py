#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Verify the immutable PR4319 bundle and its actual container load path."""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import json
import os
import platform
import socket
import stat
import struct
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

MODULE_FILENAME = "fused_moe_trtllm_sm100.so"
SOURCE_FILENAME = "trtllm_fused_moe_kernel_launcher.cu"
DEFAULT_BUNDLE = Path("/flint-flashinfer-artifact")
SITE_PACKAGES = Path("/opt/flint/envs/sglang/lib/python3.13/site-packages")
AOT_PATH = (
    SITE_PACKAGES
    / "flashinfer_jit_cache/jit_cache/fused_moe_trtllm_sm100"
    / MODULE_FILENAME
)
SOURCE_PATH = SITE_PACKAGES / "flashinfer/data/csrc" / SOURCE_FILENAME
PATCHED_SOURCE_SHA256 = (
    "7fba3ab844031058f6381ea01751a7f595fbde6931bf35346f944c306b1c0f0f"
)
UPSTREAM_PR = "https://github.com/flashinfer-ai/flashinfer/pull/4319"
EXPECTED_PACKAGES = {
    "flashinfer-python": "0.6.16.post1",
    "flashinfer-cubin": "0.6.16.post1",
    "flashinfer-jit-cache": "0.6.16.post1+cu130",
    "torch": "2.11.0+cu130",
    "sglang": "0.5.17.dev30+g3003d70f6",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def require_sha256(value, label):
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"Invalid SHA256 pin: {label}",
    )
    return value


def read_verified_file(path, expected_sha256, *, capture=False):
    """Hash a regular, unredirected file and detect changes during the read."""
    path = Path(path)
    require_sha256(expected_sha256, str(path))
    require(
        path.is_absolute() and not path.is_symlink() and path.resolve() == path,
        f"Redirected artifact path: {path}",
    )
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode), f"Artifact is not a regular file: {path}")
        digest, pieces = hashlib.sha256(), []
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            if capture:
                pieces.append(chunk)
        after = os.fstat(stream.fileno())
        current = path.stat()
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    require(
        identity(before) == identity(after) == identity(current),
        f"Artifact changed while being verified: {path}",
    )
    require(digest.hexdigest() == expected_sha256, f"SHA256 mismatch: {path}")
    return b"".join(pieces) if capture else None


def verify_elf(path):
    with Path(path).open("rb") as stream:
        header = stream.read(20)
    require(
        len(header) == 20
        and header[:6] == b"\x7fELF\x02\x01"
        and struct.unpack("<H", header[18:20])[0] == 183,
        "Patched library is not little-endian AArch64 ELF64",
    )


def verify_bundle(bundle, manifest_sha256):
    """Validate bundle provenance without importing installed GPU packages."""
    bundle = Path(bundle).absolute()
    require(
        bundle.resolve() == bundle and bundle.is_dir(),
        "Bundle directory is redirected or missing",
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(
        read_verified_file(manifest_path, manifest_sha256, capture=True)
    )
    require(
        manifest.get("schema_version") == 1 and manifest.get("status") == "validated",
        "Bundle manifest is not schema 1 / validated",
    )
    require(
        manifest.get("upstream_pr") == UPSTREAM_PR,
        "Bundle names another upstream patch",
    )
    require(
        manifest.get("packages") == EXPECTED_PACKAGES,
        "Bundle package pins differ from the reviewed runtime",
    )
    filenames = {
        "library": MODULE_FILENAME,
        "source": SOURCE_FILENAME,
        "validation": "validation-receipt.json",
        "build": "build-receipt.json",
    }
    records, payloads = {}, {}
    for role, filename in filenames.items():
        entry = manifest.get(role)
        require(
            isinstance(entry, dict) and entry.get("filename") == filename,
            f"Unexpected bundle filename: {role}",
        )
        expected = require_sha256(entry.get("sha256"), role)
        records[role] = {"path": str(bundle / filename), "sha256": expected}
        payloads[role] = read_verified_file(
            bundle / filename, expected, capture=role != "library"
        )
    require(
        manifest["source"]["sha256"] == PATCHED_SOURCE_SHA256,
        "Bundle source is not the exact reviewed PR4319 backport",
    )
    verify_elf(bundle / MODULE_FILENAME)
    library_sha256 = manifest["library"]["sha256"]
    build, validation = (
        json.loads(payloads["build"]),
        json.loads(payloads["validation"]),
    )
    require(
        build.get("status") == "built"
        and build.get("library_sha256") == library_sha256,
        "Build receipt does not identify this successfully built library",
    )
    require(
        build.get("source_sha256") == PATCHED_SOURCE_SHA256
        and build.get("library_machine") == "AArch64",
        "Build source or architecture pin differs",
    )
    require(
        validation.get("status") == "pass"
        and validation.get("all_cases_finished") is True
        and validation.get("full_engine_patched_status") == "pass",
        "GPU validation or full-engine validation did not pass completely",
    )
    require(
        validation.get("library_sha256") == library_sha256
        and validation.get("patched_source_sha256") == PATCHED_SOURCE_SHA256,
        "GPU validation refers to another library or source",
    )
    cases = validation.get("cases")
    require(
        isinstance(cases, list) and len(cases) == 3,
        "All three GPU validation cases are required",
    )
    observed = set()
    for case in cases:
        require(
            isinstance(case, dict)
            and type(case.get("pdl")) is int
            and type(case.get("memcheck")) is bool,
            "Invalid GPU validation case identity",
        )
        key = (case["pdl"], case["memcheck"])
        require(key not in observed, "Duplicate GPU validation case")
        observed.add(key)
        expected_calls = 8 if case["memcheck"] else 480
        require(
            case.get("status") == "pass"
            and type(case.get("exit_code")) is int
            and case["exit_code"] == 0
            and case.get("expected_calls") == expected_calls,
            "A required GPU validation case did not pass its complete workload",
        )
    require(
        observed == {(1, False), (0, False), (1, True)},
        "GPU validation case coverage differs",
    )
    return {
        "status": "pass",
        "manifest_sha256": manifest_sha256,
        "bundle": str(bundle),
        "library_sha256": library_sha256,
        "source_sha256": PATCHED_SOURCE_SHA256,
        "packages": dict(EXPECTED_PACKAGES),
        "files": records,
    }


def verify_loaded_module(maps_text, target, expected_sha256):
    """Check mapped path and inode, then hash the file backing the mapping."""
    target = Path(target)
    mappings = []
    for line in maps_text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and MODULE_FILENAME in fields[5]:
            mappings.append(fields)
    require(mappings, "FlashInfer MoE module was not actually mapped into this process")
    require(
        {entry[5] for entry in mappings} == {str(target)},
        "Unexpected, duplicate, or deleted FlashInfer MoE library mapping",
    )
    current = target.stat()
    for entry in mappings:
        major, minor = (int(piece, 16) for piece in entry[3].split(":"))
        require(
            (major, minor, int(entry[4]))
            == (os.major(current.st_dev), os.minor(current.st_dev), current.st_ino),
            "Loaded FlashInfer module inode differs from the verified runtime artifact",
        )
    read_verified_file(target, expected_sha256)
    return {
        "path": str(target),
        "sha256": expected_sha256,
        "device": current.st_dev,
        "inode": current.st_ino,
    }


def load_moe_module():
    # Keep GPU package imports after all bundle/runtime file and version checks.
    core = importlib.import_module("flashinfer.fused_moe.core")
    return core.get_trtllm_moe_sm100_module()


def verify_runtime(
    bundle,
    manifest_sha256,
    *,
    load=False,
    version_reader=None,
    loader=None,
    maps_reader=None,
    aot_path=AOT_PATH,
    source_path=SOURCE_PATH,
    machine_reader=None,
):
    receipt = verify_bundle(bundle, manifest_sha256)
    machine = (machine_reader or platform.machine)().lower()
    require(machine == "aarch64", "Training runtime must be AArch64")
    versions = {
        name: (version_reader or importlib.metadata.version)(name)
        for name in EXPECTED_PACKAGES
    }
    require(
        versions == EXPECTED_PACKAGES,
        "Installed package versions differ from the reviewed bundle",
    )
    read_verified_file(Path(source_path), PATCHED_SOURCE_SHA256)
    read_verified_file(Path(aot_path), receipt["library_sha256"])
    receipt.update(
        {
            "runtime_packages": versions,
            "runtime_machine": machine,
            "aot_path": str(aot_path),
            "source_path": str(source_path),
            "module_loaded": False,
        }
    )
    if load:
        (loader or load_moe_module)()
        maps_text = (maps_reader or (lambda: Path("/proc/self/maps").read_text()))()
        receipt["loaded_module"] = verify_loaded_module(
            maps_text, aot_path, receipt["library_sha256"]
        )
        receipt["module_loaded"] = True
    return receipt


def write_receipt(directory, receipt):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    enriched = dict(
        receipt,
        hostname=socket.gethostname(),
        pid=os.getpid(),
        verified_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    phase = "loaded" if receipt.get("module_loaded") else "static"
    destination = (
        directory / f"flashinfer-{phase}-{socket.gethostname()}-{os.getpid()}.json"
    )
    descriptor, temporary = tempfile.mkstemp(prefix=".flashinfer-", dir=directory)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(enriched, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # link gives an atomic create that cannot overwrite another receipt.
        os.link(temporary, destination)
    finally:
        os.unlink(temporary)
    return destination


_DEFERRED_STATE = None


def install_deferred_verification(bundle, manifest_sha256, receipt_dir):
    """Check static pins now; audit the first real MoE load without starting CUDA.

    A scoped sitecustomize installs this hook again in Python spawn/exec children.
    The loaded-module guard also tracks PID changes in fork children.
    """
    global _DEFERRED_STATE
    identity = (str(Path(bundle)), manifest_sha256, str(Path(receipt_dir)))
    if _DEFERRED_STATE is not None:
        require(
            _DEFERRED_STATE["identity"] == identity,
            "Conflicting deferred FlashInfer bundle pins",
        )
        return _DEFERRED_STATE["receipt"]
    receipt = verify_runtime(bundle, manifest_sha256, load=False)
    write_receipt(receipt_dir, receipt)
    state = {
        "identity": identity,
        "receipt": receipt,
        "verified_pid": None,
        "lock": threading.Lock(),
    }

    def reset_after_fork():
        state["verified_pid"] = None
        state["lock"] = threading.Lock()

    if hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=reset_after_fork)

    def wrap_module(module):
        original = getattr(module, "get_trtllm_moe_sm100_module", None)
        require(
            callable(original), "Pinned FlashInfer module lacks its reviewed MoE loader"
        )
        require(
            getattr(original, "_flint_verified_loader", False) is not True,
            "FlashInfer loader already has an unknown guard",
        )

        @functools.wraps(original)
        def guarded_loader(*args, **kwargs):
            result = original(*args, **kwargs)
            pid = os.getpid()
            if state["verified_pid"] != pid:
                with state["lock"]:
                    if state["verified_pid"] != pid:
                        loaded = verify_loaded_module(
                            Path("/proc/self/maps").read_text(),
                            AOT_PATH,
                            receipt["library_sha256"],
                        )
                        write_receipt(
                            receipt_dir,
                            dict(receipt, module_loaded=True, loaded_module=loaded),
                        )
                        state["verified_pid"] = pid
            return result

        guarded_loader._flint_verified_loader = True
        for name in ("cache_info", "cache_clear", "cache_parameters"):
            if hasattr(original, name):
                setattr(guarded_loader, name, getattr(original, name))
        module.get_trtllm_moe_sm100_module = guarded_loader

    class GuardedLoader(importlib.abc.Loader):
        def __init__(self, delegate):
            self.delegate = delegate

        def create_module(self, specification):
            create = getattr(self.delegate, "create_module", None)
            return create(specification) if create else None

        def exec_module(self, module):
            self.delegate.exec_module(module)
            wrap_module(module)

        def __getattr__(self, name):
            return getattr(self.delegate, name)

    class GuardedFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "flashinfer.fused_moe.core":
                return None
            specification = importlib.machinery.PathFinder.find_spec(
                fullname, path, target
            )
            require(
                specification is not None and specification.loader is not None,
                "Cannot find the pinned FlashInfer MoE implementation",
            )
            specification.loader = GuardedLoader(specification.loader)
            return specification

    if "flashinfer.fused_moe.core" in sys.modules:
        wrap_module(sys.modules["flashinfer.fused_moe.core"])
    else:
        finder = GuardedFinder()
        sys.meta_path.insert(0, finder)
        state["finder"] = finder
    _DEFERRED_STATE = state
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--bundle-only", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--receipt-dir", type=Path)
    args = parser.parse_args(argv)
    if args.bundle_only and args.load:
        parser.error("--bundle-only and --load are mutually exclusive")
    if not args.bundle_only and args.receipt_dir is None:
        parser.error("Runtime verification requires --receipt-dir")
    if args.bundle_only:
        receipt = verify_bundle(args.bundle, args.manifest_sha256)
    else:
        receipt = verify_runtime(args.bundle, args.manifest_sha256, load=args.load)
    destination = write_receipt(args.receipt_dir, receipt) if args.receipt_dir else None
    print(
        json.dumps(
            {
                "status": "pass",
                "bundle_only": args.bundle_only,
                "module_loaded": receipt.get("module_loaded", False),
                "receipt_path": str(destination) if destination else None,
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(
            f"FlashInfer runtime verification failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        sys.exit(1)
