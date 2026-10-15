"""MSC publication tests using its POSIX provider (no cloud credentials)."""

import json
import uuid

import pytest

from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.utils.checkpoint_storage import (
    enable_msc,
    load_checkpoint_object,
    open_path,
    path_exists,
    save_checkpoint_object,
)


@pytest.fixture
def manager(tmp_path, monkeypatch):
    pytest.importorskip("multistorageclient")
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
    assert not enable_msc().os.path.exists(checkpoint + "/_SUCCESS")
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
