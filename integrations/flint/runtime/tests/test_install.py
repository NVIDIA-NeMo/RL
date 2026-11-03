"""Exercise exact-source overlay guards without private model files or GPUs."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from install import prepare_patch
from launch_grpo import actor_executable_mapping


class InstallTests(unittest.TestCase):
    def test_patch_is_verified_before_install_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.py"
            source.write_bytes(b"before\n")
            patch = root / "change.patch"
            patch.write_text(
                "--- a/model.py\n+++ b/model.py\n@@ -1 +1 @@\n-before\n+after\n"
            )
            pins = {
                "model.py": tuple(
                    hashlib.sha256(v).hexdigest() for v in (b"before\n", b"after\n")
                )
            }
            plan = prepare_patch(root, patch, pins)
            self.assertEqual(source.read_bytes(), b"before\n")
            self.assertEqual(plan[source], b"after\n")
            source.write_bytes(plan[source])
            self.assertEqual(prepare_patch(root, patch, pins), {})
            source.write_bytes(b"unknown\n")
            with self.assertRaisesRegex(RuntimeError, "Unknown"):
                prepare_patch(root, patch, pins)

    def test_router_and_generation_use_the_verified_serving_profile(self) -> None:
        mapping = actor_executable_mapping()
        router = mapping["nemo_rl.models.generation.sglang.sglang_router.RouterActor"]
        worker = mapping[
            "nemo_rl.models.generation.sglang.sglang_worker.SGLangGenerationWorker"
        ]
        self.assertEqual(router, worker)
        self.assertTrue(router.endswith("python-sglang.sh"))
        self.assertNotEqual(
            router,
            mapping[
                "nemo_rl.models.policy.workers.automodel_policy_worker.AutomodelPolicyWorker"
            ],
        )
