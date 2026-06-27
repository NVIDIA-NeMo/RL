# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# This directory contains NeMo-RL configuration files for nemo_gym environments.
# It is NOT the nemo_gym library — that lives in 3rdparty/Gym-workspace/Gym/.
#
# Problem: when Python runs `examples/run_grpo.py`, it inserts `examples/` into
# sys.path[0]. Python's PathFinder then finds this directory as a namespace
# package (no __init__.py → __file__ = None → "(unknown location)"), shadowing
# the real nemo_gym library before the editable-install finder can reach it.
#
# Fix: temporarily filter `examples/` out of sys.path, import the real nemo_gym,
# then replace this placeholder module in sys.modules with the real one.

import os
import sys

_self_dir = os.path.dirname(os.path.abspath(__file__))  # .../examples/nemo_gym
_examples_dir = os.path.abspath(os.path.dirname(_self_dir))  # .../examples

# Build a filtered sys.path that excludes examples/ so PathFinder won't loop
# back to this file when we re-import nemo_gym below.
_saved_path = list(sys.path)
sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) != _examples_dir]

# Remove the in-progress (fake) module so importlib can find the real package.
sys.modules.pop("nemo_gym", None)

try:
    import nemo_gym as _real_nemo_gym  # finds the real Gym library
except Exception:
    sys.path[:] = _saved_path  # restore on failure so callers get a clean state
    raise

# Preserve any new sys.path entries added by the real __init__.py (e.g. PARENT_DIR).
_added = [p for p in sys.path if p not in set(_saved_path)]
sys.path[:] = _saved_path + _added

# Replace ourselves with the real module for this and all future imports.
sys.modules["nemo_gym"] = _real_nemo_gym
globals().update(vars(_real_nemo_gym))
