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
"""NeMo RL Trainers Module.

This module provides base trainer classes for all NeMo RL training algorithms.
Algorithm-specific trainers should inherit from BaseTrainer.

Example:
    >>> from nemo_rl.trainers import BaseTrainer
    >>> 
    >>> class MyTrainer(BaseTrainer):
    ...     def _train_step(self, batch):
    ...         return {"loss": compute_loss(batch)}
    ...     
    ...     def _compute_loss(self, batch, outputs):
    ...         return loss_fn(batch, outputs)
    >>> 
    >>> trainer = MyTrainer(config)
    >>> trainer.fit(datamodule)

Callbacks:
    >>> from nemo_rl.trainers import Callback, CheckpointCallback
    >>> 
    >>> class MyCallback(Callback):
    ...     def on_epoch_end(self, trainer, epoch, logs):
    ...         print(f"Epoch {epoch}: {logs['loss']:.4f}")
    >>> 
    >>> trainer.fit(
    ...     datamodule,
    ...     callbacks=[MyCallback(), CheckpointCallback(every_n_epochs=1)]
    ... )
"""

from nemo_rl.trainers.base import (
    BaseTrainer,
    TrainerState,
    TrainingResult,
    ValidationResult,
)
from nemo_rl.trainers.callbacks import (
    Callback,
    CallbackList,
    CheckpointCallback,
    EarlyStoppingCallback,
    LambdaCallback,
    LoggingCallback,
    ProgressCallback,
)
from nemo_rl.trainers.validation import (
    ValidationConfig,
    ValidationResult as ValidationRunnerResult,
    ValidationRunner,
    create_validation_runner,
)

__all__ = [
    # Trainer classes
    "BaseTrainer",
    "TrainerState",
    "TrainingResult",
    "ValidationResult",
    # Validation runner
    "ValidationRunner",
    "ValidationConfig",
    "ValidationRunnerResult",
    "create_validation_runner",
    # Callback classes
    "Callback",
    "CallbackList",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "LambdaCallback",
    "LoggingCallback",
    "ProgressCallback",
]
