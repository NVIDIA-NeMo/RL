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
"""MSC publication tests using its POSIX provider (no cloud credentials)."""

import json
import uuid

import multistorageclient as msc
import pytest

from nemo_rl.utils.checkpoint import (
    CheckpointManager,
    _load_checkpoint_history,
    validate_warm_start_checkpoint,
)
from nemo_rl.utils.path_utils import (
    as_path,
    listdir,
    load_checkpoint_object,
    maybe_enable_msc,
    open_path,
    path_exists,
    rmtree,
    save_checkpoint_object,
)


@pytest.fixture
def manager(tmp_path, monkeypatch):
    pytest.importorskip("megatron.core")
    profile = "test-" + uuid.uuid4().hex
    config = tmp_path / "msc.json"
    config.write_text(
        json.dumps(
            {
                "profiles": {
                    profile: {
                        "storage_provider": {
                            "type": "file",
                            "options": {"base_path": str(tmp_path / "objects")},
                        }
                    }
                }
            }
        )
    )
    monkeypatch.setenv("MSC_CONFIG", str(config))
    (tmp_path / "objects").mkdir()
    with CheckpointManager(
        {
            "checkpoint_dir": f"msc://{profile}/run",
            "metric_name": None,
            "higher_is_better": True,
            "keep_top_k": None,
            "save_period": 1,
            "save_optimizer": False,
            "save_data_plane": False,
        }
    ) as checkpointer:
        yield checkpointer


def test_unfinished_checkpoint_is_not_discoverable(manager):
    checkpoint = manager.init_tmp_checkpoint(1, {"step": 1})
    assert manager.get_latest_checkpoint_path() is None
    manager.finalize_checkpoint(checkpoint)
    assert manager.get_latest_checkpoint_path() == checkpoint
    unfinished = manager.init_tmp_checkpoint(2, {"step": 2})
    assert unfinished != checkpoint
    assert manager.get_latest_checkpoint_path() == checkpoint


def test_failed_async_writer_never_publishes(manager):
    checkpoint = manager.init_tmp_checkpoint(1, {"step": 1})

    def fail():
        raise OSError("simulated shard write failure")

    manager.begin_finalization(checkpoint, wait_fn=fail)
    with pytest.raises(RuntimeError, match="finalization failed"):
        manager.finalize_pending()
    assert not msc.os.path.exists(checkpoint + "/_SUCCESS")
    assert manager.get_latest_checkpoint_path() is None


def test_existing_checkpoint_cannot_be_overwritten(manager):
    checkpoint = manager.init_tmp_checkpoint(1, {"step": 1})
    manager.finalize_checkpoint(checkpoint)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        manager.init_tmp_checkpoint(1, {"step": 100})


def test_controller_state_round_trips_through_msc(manager):
    base = str(manager.checkpoint_dir) + "/step_1"
    save_checkpoint_object({"samples_yielded": 7}, base + "/train_dataloader.pt")
    assert load_checkpoint_object(base + "/train_dataloader.pt") == {
        "samples_yielded": 7
    }
    # Async GRPO restores replay-buffer/rollout pickles with weights_only=False.
    assert load_checkpoint_object(
        base + "/train_dataloader.pt", weights_only=False
    ) == {"samples_yielded": 7}


def test_open_path_and_path_exists_through_msc(manager):
    status = str(manager.checkpoint_dir) + "/latest_checkpoint_status.json"
    assert not path_exists(status)
    with open_path(status, "w") as f:
        json.dump({"last_checkpoint_step": 5}, f)
    assert path_exists(status)
    with open_path(status) as f:
        assert json.load(f) == {"last_checkpoint_step": 5}


def test_plain_paths_bypass_msc(tmp_path):
    state = tmp_path / "train_dataloader.pt"
    save_checkpoint_object({"samples_yielded": 3}, state)
    assert load_checkpoint_object(state) == {"samples_yielded": 3}
    assert path_exists(state)
    with open_path(tmp_path / "x.json", "w") as f:
        f.write("{}")
    assert not path_exists(tmp_path / "missing.json")


def test_save_dataloader_state_uses_shared_path_for_local_and_msc(tmp_path, manager):
    save_dataloader_state = pytest.importorskip(
        "nemo_rl.data.utils"
    ).save_dataloader_state
    save_dataloader_state({"samples_yielded": 1}, str(tmp_path), suffix="_math")
    assert load_checkpoint_object(tmp_path / "train_dataloader_math.pt") == {
        "samples_yielded": 1
    }

    base = str(manager.checkpoint_dir) + "/step_1"
    save_dataloader_state({"samples_yielded": 2}, base)
    assert load_checkpoint_object(base + "/train_dataloader.pt") == {
        "samples_yielded": 2
    }


def test_maybe_enable_msc_only_enables_for_msc_urls(tmp_path, manager):
    from megatron.core.msc_utils import MultiStorageClientFeature

    MultiStorageClientFeature.disable()
    maybe_enable_msc(None, str(tmp_path), "")
    assert not MultiStorageClientFeature.is_enabled()
    maybe_enable_msc(str(tmp_path), str(manager.checkpoint_dir))
    assert MultiStorageClientFeature.is_enabled()


def test_interrupted_step_can_be_saved_again_after_resume(manager):
    first = manager.init_tmp_checkpoint(1, {"step": 1})
    stale = first + "/policy/weights/shard.bin"
    with msc.open(stale, "w") as f:
        f.write("partial write from a preempted run")
    # No _SUCCESS was published, so a resumed run may re-save the same step.
    again = manager.init_tmp_checkpoint(1, {"step": 1})
    assert again == first
    assert not msc.os.path.exists(stale)
    manager.finalize_checkpoint(again)
    assert manager.get_latest_checkpoint_path() == again


def test_history_lists_only_published_steps_with_their_info(manager):
    for step in (1, 2):
        manager.finalize_checkpoint(manager.init_tmp_checkpoint(step, {"step": step}))
    manager.init_tmp_checkpoint(3, {"step": 3})  # never published
    history = _load_checkpoint_history(manager._storage)
    assert sorted((step, info["step"]) for step, _, info in history) == [(1, 1), (2, 2)]
    assert manager.get_latest_checkpoint_path().endswith("/step_2")


def test_rmtree_and_as_path_keep_msc_urls_intact(manager):
    checkpoint = manager.init_tmp_checkpoint(1, {"step": 1})
    assert str(as_path(checkpoint)) == checkpoint
    rmtree(checkpoint)
    assert not path_exists(checkpoint + "/training_info.json")


def test_warm_start_validation_accepts_msc_urls(manager):
    checkpoint = manager.init_tmp_checkpoint(1, {"step": 1})
    with open_path(checkpoint + "/value/weights/shard.bin", "w") as f:
        f.write("weights")
    validate_warm_start_checkpoint(checkpoint, model_component="value")
    with pytest.raises(ValueError, match="no policy/weights subtree"):
        validate_warm_start_checkpoint(checkpoint, model_component="policy")


def test_listdir_lists_msc_children(manager):
    checkpoint = manager.init_tmp_checkpoint(1, {"step": 1})
    with open_path(checkpoint + "/policy/weights/shard.bin", "w") as f:
        f.write("weights")
    assert sorted(listdir(checkpoint)) == ["policy", "training_info.json"]


def test_get_resume_paths_enables_msc_before_megatron_load(manager, monkeypatch):
    from megatron.core.msc_utils import MultiStorageClientFeature

    checkpoint = manager.init_tmp_checkpoint(1, {"step": 1})
    with open_path(checkpoint + "/value/weights/iter_0000000/metadata.json", "w") as f:
        f.write("{}")
    MultiStorageClientFeature.disable()
    enabled_at_load = []

    def fake_load(iteration_dir):
        enabled_at_load.append(MultiStorageClientFeature.is_enabled())
        return {"optimizer": {}}

    monkeypatch.setattr(
        "nemo_rl.utils.checkpoint._load_megatron_common_state_dict", fake_load
    )
    CheckpointManager.get_resume_paths(checkpoint, model_component="value")
    assert enabled_at_load == [True]
