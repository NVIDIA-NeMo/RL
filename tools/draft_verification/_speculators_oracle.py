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
"""Import bootstrap for the speculators reference ("oracle") source tree.

The vendored draft trainers in ``nemo_rl/models/automodel/draft/`` are copied
from vllm-project/speculators @ 0b08a89. The verification scripts in this
directory import that original source and compare it numerically against the
vendored code. The speculators package is not installed in the training
container, so this module makes a source checkout importable:

- ``sys.path`` gains ``<speculators>/src``.
- ``importlib.metadata.version("speculators")`` is patched (the package
  ``__init__`` requires installed dist metadata).
- Unrelated heavy dependencies pulled in by ``speculators/__init__``
  (``openai``, ``hs_connectors``) are auto-stubbed: any attribute access on
  them returns an empty placeholder class. Nothing the oracle math touches
  lives in those modules.
"""

import importlib.abc
import importlib.machinery
import importlib.metadata
import sys
import types

DEFAULT_SPECULATORS_PATH = (
    "/lustre/fsw/portfolios/coreai/users/yuekaiz/speculative/speculators"
)

_STUB_ROOTS = ("openai", "hs_connectors")


class _AutoStubModule(types.ModuleType):
    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {})


class _AutoStubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        module = _AutoStubModule(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module):
        pass


class _AutoStubFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in _STUB_ROOTS:
            return importlib.machinery.ModuleSpec(
                fullname, _AutoStubLoader(), is_package=True
            )
        return None


def bootstrap_speculators(speculators_repo: str = DEFAULT_SPECULATORS_PATH) -> None:
    """Make the speculators source checkout importable in this process."""
    src = f"{speculators_repo}/src"
    if src not in sys.path:
        sys.path.insert(0, src)

    original_version = importlib.metadata.version

    def _version(name: str) -> str:
        try:
            return original_version(name)
        except importlib.metadata.PackageNotFoundError:
            if name == "speculators":
                return "0.0.0.dev0"
            raise

    importlib.metadata.version = _version

    if not any(isinstance(f, _AutoStubFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _AutoStubFinder())
