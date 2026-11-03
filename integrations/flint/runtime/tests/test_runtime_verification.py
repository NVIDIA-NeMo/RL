"""CPU-only rejection and Python worker forwarding tests; no GPU claim."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import launch_verified_sglang as launch
import verify_runtime as verify


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


class BundleFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir="/tmp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        self.source = b"test physical-capacity source fixture\n"
        self.source_pin = sha(self.source)
        self.source_patch = mock.patch.object(
            verify, "PATCHED_SOURCE_SHA256", self.source_pin
        )
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)
        elf = bytearray(64)
        elf[:6] = b"\x7fELF\x02\x01"
        elf[18:20] = struct.pack("<H", 183)
        self.library = bytes(elf)
        self.library_pin = sha(self.library)
        self.build = {
            "status": "built",
            "library_sha256": self.library_pin,
            "source_sha256": self.source_pin,
            "library_machine": "AArch64",
        }
        self.validation = {
            "status": "pass",
            "all_cases_finished": True,
            "full_engine_patched_status": "pass",
            "library_sha256": self.library_pin,
            "patched_source_sha256": self.source_pin,
            "cases": [
                {
                    "pdl": pdl,
                    "memcheck": memcheck,
                    "status": "pass",
                    "exit_code": 0,
                    "expected_calls": 8 if memcheck else 480,
                }
                for pdl, memcheck in [(1, False), (0, False), (1, True)]
            ],
        }
        self.rebuild()

    def rebuild(self):
        payloads = {
            "library": (verify.MODULE_FILENAME, self.library),
            "source": (verify.SOURCE_FILENAME, self.source),
            "build": ("build-receipt.json", json.dumps(self.build).encode()),
            "validation": (
                "validation-receipt.json",
                json.dumps(self.validation).encode(),
            ),
        }
        self.manifest = {
            "schema_version": 1,
            "status": "validated",
            "upstream_pr": verify.UPSTREAM_PR,
            "packages": dict(verify.EXPECTED_PACKAGES),
        }
        for role, (filename, payload) in payloads.items():
            (self.bundle / filename).write_bytes(payload)
            self.manifest[role] = {"filename": filename, "sha256": sha(payload)}
        self.save_manifest()

    def save_manifest(self):
        payload = json.dumps(self.manifest).encode()
        (self.bundle / "manifest.json").write_bytes(payload)
        self.manifest_pin = sha(payload)

    def runtime_arguments(self):
        aot, source = self.root / "runtime.so", self.root / "runtime.cu"
        aot.write_bytes(self.library)
        source.write_bytes(self.source)
        return {
            "aot_path": aot,
            "source_path": source,
            "machine_reader": lambda: "aarch64",
            "version_reader": verify.EXPECTED_PACKAGES.__getitem__,
        }

    @staticmethod
    def maps_for(path, inode=None):
        record = Path(path).stat()
        return (
            f"1000-2000 r-xp 00000000 {os.major(record.st_dev):02x}:{os.minor(record.st_dev):02x} "
            f"{record.st_ino if inode is None else inode} {path}\n"
        )

    def test_valid_bundle_does_not_import_installed_gpu_packages(self):
        with mock.patch.object(
            verify.importlib.metadata,
            "version",
            side_effect=AssertionError("GPU metadata access"),
        ):
            result = verify.verify_bundle(self.bundle, self.manifest_pin)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    verify.main(
                        [
                            "--bundle",
                            str(self.bundle),
                            "--manifest-sha256",
                            self.manifest_pin,
                            "--bundle-only",
                        ]
                    ),
                    0,
                )
        self.assertEqual(result["library_sha256"], self.library_pin)

    def test_manifest_and_binary_tampering_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
            verify.verify_bundle(self.bundle, "0" * 64)
        (self.bundle / verify.MODULE_FILENAME).write_bytes(self.library + b"changed")
        with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
            verify.verify_bundle(self.bundle, self.manifest_pin)

    def test_manifest_cannot_redirect_library(self):
        self.manifest["library"]["filename"] = "../unreviewed.so"
        self.save_manifest()
        with self.assertRaisesRegex(RuntimeError, "filename"):
            verify.verify_bundle(self.bundle, self.manifest_pin)

    def test_source_pin_is_independent_of_manifest(self):
        self.source = b"different source"
        self.rebuild()
        with self.assertRaisesRegex(RuntimeError, "exact reviewed"):
            verify.verify_bundle(self.bundle, self.manifest_pin)

    def test_aarch64_binary_required(self):
        value = bytearray(self.library)
        value[18:20] = struct.pack("<H", 62)
        self.library = bytes(value)
        self.library_pin = sha(self.library)
        self.build["library_sha256"] = self.validation["library_sha256"] = (
            self.library_pin
        )
        self.rebuild()
        with self.assertRaisesRegex(RuntimeError, "AArch64 ELF64"):
            verify.verify_bundle(self.bundle, self.manifest_pin)

    def test_partial_full_engine_validation_rejected(self):
        for field, value in [
            ("all_cases_finished", False),
            ("full_engine_patched_status", "fail"),
        ]:
            with self.subTest(field=field):
                self.validation[field] = value
                self.rebuild()
                with self.assertRaisesRegex(RuntimeError, "did not pass completely"):
                    verify.verify_bundle(self.bundle, self.manifest_pin)
                self.validation[field] = (
                    True if field == "all_cases_finished" else "pass"
                )

    def test_duplicate_or_incomplete_case_coverage_rejected(self):
        self.validation["cases"][1] = dict(self.validation["cases"][0])
        self.rebuild()
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            verify.verify_bundle(self.bundle, self.manifest_pin)
        self.validation["cases"] = self.validation["cases"][:2]
        self.rebuild()
        with self.assertRaisesRegex(RuntimeError, "three"):
            verify.verify_bundle(self.bundle, self.manifest_pin)

    def test_case_failure_or_short_workload_rejected(self):
        self.validation["cases"][0]["expected_calls"] = 1
        self.rebuild()
        with self.assertRaisesRegex(RuntimeError, "complete workload"):
            verify.verify_bundle(self.bundle, self.manifest_pin)

    def test_receipts_must_bind_same_artifact(self):
        self.validation["library_sha256"] = "0" * 64
        self.rebuild()
        with self.assertRaisesRegex(RuntimeError, "another library"):
            verify.verify_bundle(self.bundle, self.manifest_pin)

    def test_package_mismatch_rejected_before_loader(self):
        arguments = self.runtime_arguments()
        arguments["version_reader"] = lambda name: "unknown"
        loader = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "Installed package"):
            verify.verify_runtime(
                self.bundle, self.manifest_pin, load=True, loader=loader, **arguments
            )
        loader.assert_not_called()

    def test_actual_loaded_path_and_inode_verified(self):
        arguments = self.runtime_arguments()
        # Give the simulated runtime target the real module basename, as /proc maps does.
        aot = self.root / verify.MODULE_FILENAME
        aot.write_bytes(self.library)
        arguments["aot_path"] = aot
        loader = mock.Mock()
        result = verify.verify_runtime(
            self.bundle,
            self.manifest_pin,
            load=True,
            loader=loader,
            maps_reader=lambda: self.maps_for(aot),
            **arguments,
        )
        self.assertTrue(result["module_loaded"])
        loader.assert_called_once_with()
        with self.assertRaisesRegex(RuntimeError, "inode"):
            verify.verify_loaded_module(
                self.maps_for(aot, inode=0), aot, self.library_pin
            )
        with self.assertRaisesRegex(RuntimeError, "mapping"):
            verify.verify_loaded_module(
                self.maps_for(aot).rstrip() + " (deleted)\n", aot, self.library_pin
            )
        alternate = self.root / "other" / verify.MODULE_FILENAME
        alternate.parent.mkdir()
        alternate.write_bytes(self.library)
        with self.assertRaisesRegex(RuntimeError, "mapping"):
            verify.verify_loaded_module(
                self.maps_for(aot) + self.maps_for(alternate), aot, self.library_pin
            )

    def test_unknown_runtime_source_rejected(self):
        arguments = self.runtime_arguments()
        arguments["source_path"].write_bytes(b"unknown source")
        with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
            verify.verify_runtime(self.bundle, self.manifest_pin, **arguments)


class WorkerTests(unittest.TestCase):
    def test_parser_accepts_ray_script_and_preserves_arguments(self):
        result = launch.parse_python_invocation(
            ["-u", "-B", "/ray/default_worker.py", "--node-ip", "1.2.3.4"]
        )
        self.assertEqual(
            (result.mode, result.target, result.arguments),
            ("script", "/ray/default_worker.py", ("--node-ip", "1.2.3.4")),
        )
        self.assertTrue(result.unbuffered and result.dont_write_bytecode)
        self.assertEqual(
            launch.parse_python_invocation(["-m", "multiprocessing.spawn"]).mode,
            "module",
        )
        self.assertEqual(launch.parse_python_invocation(["-c", ""]).mode, "command")

    def test_parser_rejects_unsupported_interpreter_modes(self):
        for arguments in [
            [],
            ["-"],
            ["-I", "worker.py"],
            ["-X", "dev", "worker.py"],
            ["-m"],
            ["-c"],
        ]:
            with self.subTest(arguments=arguments), self.assertRaises(RuntimeError):
                launch.parse_python_invocation(arguments)

    def test_script_module_command_forward_real_argv_and_main(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory).resolve()
            script = root / "forward_fixture.py"
            script.write_text(
                "import json, sys\nfrom pathlib import Path\n"
                "Path(sys.argv[1]).write_text(json.dumps({'argv':sys.argv,'name':__name__,'path':sys.path[0]}))\n"
            )
            for mode, target in [
                ("script", str(script)),
                ("module", "forward_fixture"),
                ("command", script.read_text()),
            ]:
                destination = root / (mode + ".json")
                original_main = sys.modules["__main__"]
                with (
                    mock.patch.object(sys, "argv", list(sys.argv)),
                    mock.patch.object(sys, "path", ["", str(root), *sys.path]),
                ):
                    try:
                        launch.forward_python_invocation(
                            launch.PythonInvocation(
                                mode, target, (str(destination), "--flag")
                            )
                        )
                    finally:
                        sys.modules["__main__"] = original_main
                result = json.loads(destination.read_text())
                self.assertEqual(result["argv"][1:], [str(destination), "--flag"])
                self.assertEqual(result["name"], "__main__")
                self.assertEqual(result["path"], str(root) if mode == "script" else "")
                self.assertEqual(
                    result["argv"][0], "-c" if mode == "command" else str(script)
                )

    def test_worker_exit_code_propagates(self):
        original_main = sys.modules["__main__"]
        with (
            mock.patch.object(sys, "argv", list(sys.argv)),
            mock.patch.object(sys, "path", list(sys.path)),
        ):
            try:
                with self.assertRaises(SystemExit) as raised:
                    launch.forward_python_invocation(
                        launch.PythonInvocation("command", "raise SystemExit(7)", ())
                    )
                self.assertEqual(raised.exception.code, 7)
            finally:
                sys.modules["__main__"] = original_main

    def test_deferred_guard_has_no_early_gpu_load_and_verifies_once_per_pid(self):
        module = types.ModuleType("flashinfer.fused_moe.core")
        module.get_trtllm_moe_sm100_module = original = mock.Mock(
            return_value="loaded-handle"
        )
        receipt = {"status": "pass", "library_sha256": "a" * 64, "module_loaded": False}
        with (
            mock.patch.dict(sys.modules, {module.__name__: module}),
            mock.patch.object(verify, "_DEFERRED_STATE", None),
            mock.patch.object(verify, "verify_runtime", return_value=receipt) as static,
            mock.patch.object(verify, "write_receipt") as writer,
            mock.patch.object(
                verify, "verify_loaded_module", return_value={"path": "pinned"}
            ) as loaded,
            mock.patch.object(Path, "read_text", return_value="maps"),
            mock.patch.object(os, "getpid", return_value=123) as pid,
        ):
            verify.install_deferred_verification("/bundle", "a" * 64, "/receipts")
            verify.install_deferred_verification("/bundle", "a" * 64, "/receipts")
            static.assert_called_once()
            original.assert_not_called()
            self.assertEqual(module.get_trtllm_moe_sm100_module(), "loaded-handle")
            module.get_trtllm_moe_sm100_module()
            loaded.assert_called_once()
            pid.return_value = 124
            module.get_trtllm_moe_sm100_module()
            self.assertEqual(loaded.call_count, 2)
            self.assertEqual(writer.call_count, 3)  # one static, one loaded per PID

    def test_deferred_guard_rejects_wrong_mapping_before_returning_handle(self):
        module = types.ModuleType("flashinfer.fused_moe.core")
        module.get_trtllm_moe_sm100_module = mock.Mock(return_value="must-not-escape")
        with (
            mock.patch.dict(sys.modules, {module.__name__: module}),
            mock.patch.object(verify, "_DEFERRED_STATE", None),
            mock.patch.object(
                verify, "verify_runtime", return_value={"library_sha256": "a" * 64}
            ),
            mock.patch.object(verify, "write_receipt"),
            mock.patch.object(
                verify, "verify_loaded_module", side_effect=RuntimeError("wrong module")
            ),
            mock.patch.object(Path, "read_text", return_value="maps"),
        ):
            verify.install_deferred_verification("/bundle", "a" * 64, "/receipts")
            with self.assertRaisesRegex(RuntimeError, "wrong module"):
                module.get_trtllm_moe_sm100_module()

    def test_child_startup_cannot_ignore_missing_verification_pins(self):
        environment = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(TOOLS / "sglang_verified_startup") + os.pathsep + str(TOOLS),
        )
        environment.pop("FLINT_FLASHINFER_MANIFEST_SHA256", None)
        environment.pop("FLINT_FLASHINFER_RECEIPT_DIR", None)
        result = subprocess.run(
            [sys.executable, "-c", "print('WORKER_RAN')"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("WORKER_RAN", result.stdout)
        self.assertIn("Verified SGLang startup failed", result.stderr)


if __name__ == "__main__":
    unittest.main()
