# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for the fastokens env-var-gated monkey-patch utility.

Two flavors of coverage:

1. Gating/idempotency/error-handling of :func:`maybe_patch_fastokens`. These use
   a fake ``fastokens`` module so they run everywhere, regardless of whether the
   real (platform-specific) wheel is installed.
2. An interchangeability check that asserts the real fastokens tokenizer produces
   byte-identical output to stock HuggingFace ``transformers``. Because
   ``fastokens.patch_transformers()`` mutates ``transformers`` globally and
   irreversibly, this runs in a subprocess so it cannot pollute the rest of the
   test session. It is skipped when the fastokens wheel is unavailable.
"""

import subprocess
import sys
import types

import pytest

import nemo_rl.utils.fastokens as fastokens_util


@pytest.fixture(autouse=True)
def _reset_fastokens_state(monkeypatch):
    """Reset module-level patch flag and env var before each test."""
    monkeypatch.setattr(fastokens_util, "_patched", False)
    monkeypatch.delenv("NRL_USE_FASTOKENS", raising=False)


def _install_fake_fastokens(monkeypatch, patch_fn):
    fake = types.ModuleType("fastokens")
    fake.patch_transformers = patch_fn
    monkeypatch.setitem(sys.modules, "fastokens", fake)
    return fake


def test_noop_when_config_disabled(monkeypatch):
    calls = []
    _install_fake_fastokens(monkeypatch, lambda: calls.append(1))

    fastokens_util.maybe_patch_fastokens(False)

    assert calls == []
    assert fastokens_util._patched is False


def test_patches_once_when_config_enabled(monkeypatch):
    calls = []
    _install_fake_fastokens(monkeypatch, lambda: calls.append(1))

    fastokens_util.maybe_patch_fastokens(True)
    fastokens_util.maybe_patch_fastokens(True)  # idempotent: second call is a no-op

    assert calls == [1]
    assert fastokens_util._patched is True


def test_env_override_forces_on_over_disabled_config(monkeypatch):
    calls = []
    _install_fake_fastokens(monkeypatch, lambda: calls.append(1))
    monkeypatch.setenv("NRL_USE_FASTOKENS", "1")

    fastokens_util.maybe_patch_fastokens(False)  # config off, env forces on

    assert calls == [1]
    assert fastokens_util._patched is True


def test_env_override_forces_off_over_enabled_config(monkeypatch):
    calls = []
    _install_fake_fastokens(monkeypatch, lambda: calls.append(1))
    monkeypatch.setenv("NRL_USE_FASTOKENS", "0")

    fastokens_util.maybe_patch_fastokens(True)  # config on, env forces off

    assert calls == []
    assert fastokens_util._patched is False


def test_missing_package_is_non_fatal(monkeypatch):
    # A ``None`` entry in sys.modules makes ``import fastokens`` raise ImportError.
    monkeypatch.setitem(sys.modules, "fastokens", None)

    fastokens_util.maybe_patch_fastokens(True)  # must not raise

    assert fastokens_util._patched is False


def test_patch_failure_is_non_fatal(monkeypatch):
    def boom():
        raise RuntimeError("kaboom")

    _install_fake_fastokens(monkeypatch, boom)

    fastokens_util.maybe_patch_fastokens(True)  # must not raise

    assert fastokens_util._patched is False


_EQUIVALENCE_SCRIPT = r"""
import sys

from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SAMPLES = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "def add(a, b):\n    return a + b  # inline comment",
    "Unicode: cafe\u0301, nai\u0308ve, \u65e5\u672c\u8a9e, emoji \U0001f680\U0001f525, math \u2211\u222b\u221a",
    "   leading and   multiple    internal   spaces\ttabs\n\nnewlines",
    "A" * 4000,
]

# Capture stock HuggingFace outputs BEFORE the (global, irreversible) patch.
tok = AutoTokenizer.from_pretrained(MODEL)
base_call = [tok(s)["input_ids"] for s in SAMPLES]
base_encode = [tok.encode(s) for s in SAMPLES]
base_batch = tok(SAMPLES, padding=True)["input_ids"]
base_decode = [tok.decode(ids) for ids in base_call]

import fastokens

fastokens.patch_transformers()

# Re-instantiate to mirror get_tokenizer(), which patches then constructs.
ftok = AutoTokenizer.from_pretrained(MODEL)
fast_call = [ftok(s)["input_ids"] for s in SAMPLES]
fast_encode = [ftok.encode(s) for s in SAMPLES]
fast_batch = ftok(SAMPLES, padding=True)["input_ids"]
fast_decode = [ftok.decode(ids) for ids in fast_call]

assert fast_call == base_call, "__call__ input_ids mismatch"
assert fast_encode == base_encode, "encode() mismatch"
assert fast_batch == base_batch, "batch encode mismatch"
assert fast_decode == base_decode, "decode() mismatch"
print("FASTOKENS_EQUIVALENCE_OK")
"""


def test_fastokens_equivalent_to_huggingface():
    """Real fastokens must be byte-identical to stock transformers (subprocess-isolated)."""
    pytest.importorskip("fastokens")

    result = subprocess.run(
        [sys.executable, "-c", _EQUIVALENCE_SCRIPT],
        capture_output=True,
        text=True,
    )

    assert "FASTOKENS_EQUIVALENCE_OK" in result.stdout, (
        f"equivalence subprocess failed (rc={result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert result.returncode == 0
