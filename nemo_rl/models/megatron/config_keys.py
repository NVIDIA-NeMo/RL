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

"""Megatron config key groups shared across driver and worker code.

Deliberately import-free: this module is imported by driver-side code
(e.g. TeacherWorkerGroup) that must not pull in megatron/transformer_engine.
"""

VLM_TOWER_OVERRIDE_KEYS = (
    "radio_force_cpe_eval_mode",
    "freeze_vision_model",
    "freeze_vision_projection",
    "freeze_sound_encoder",
    "freeze_sound_projection",
)

# Model-architecture keys that must never leak from a student config onto a
# teacher: the teacher's structure comes from its own checkpoint, or from an
# explicit per-teacher override. TeacherWorkerGroup strips these at clone time.
TEACHER_ARCHITECTURE_KEYS = VLM_TOWER_OVERRIDE_KEYS + (
    "mtp_num_layers",
    "mtp_use_repeated_layer",
    "mtp_loss_scaling_factor",
    "mtp_detach_heads",
)
