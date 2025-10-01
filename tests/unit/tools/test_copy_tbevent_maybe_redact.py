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
import filecmp
import glob
import os
import sys
import tempfile
from unittest.mock import patch

import pytest

# Make sure the script can be imported
script_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools")
)
sys.path.insert(0, script_dir)

import copy_tbevent_maybe_redact as script_under_test

# Needed for reading back and verifying hparams
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.plugins.hparams import plugin_data_pb2
from torch.utils.tensorboard import SummaryWriter


def extract_hparams_from_event_file(file_path: str) -> dict | None:
    """Extracts HParams dictionary from a single TensorBoard event file."""
    try:
        ea = event_accumulator.EventAccumulator(
            file_path,
            size_guidance=script_under_test.SIZE_GUIDANCE_META,  # Use defined size guidance
        )
        ea.Reload()  # Load the events

        # Check for HParams using the specific tag and plugin name
        if script_under_test.HPARAMS_TAG in ea.summary_metadata:
            metadata = ea.summary_metadata[script_under_test.HPARAMS_TAG]
            if (
                metadata.plugin_data
                and metadata.plugin_data.plugin_name
                == script_under_test.HPARAMS_PLUGIN_NAME
            ):
                try:
                    plugin_data_proto = plugin_data_pb2.HParamsPluginData.FromString(
                        metadata.plugin_data.content
                    )
                    if plugin_data_proto.HasField("session_start_info"):
                        # Convert the protobuf map to a standard Python dict
                        hparams_dict = {}
                        for (
                            key,
                            hparam_value,
                        ) in plugin_data_proto.session_start_info.hparams.items():
                            if hparam_value.HasField("string_value"):
                                hparams_dict[key] = hparam_value.string_value
                            elif hparam_value.HasField("number_value"):
                                hparams_dict[key] = (
                                    hparam_value.number_value
                                )  # Store as float/int
                            elif hparam_value.HasField("bool_value"):
                                hparams_dict[key] = hparam_value.bool_value
                        return hparams_dict
                    else:
                        print(
                            f"Warning: HParams data in {file_path} missing 'session_start_info' field.",
                            file=sys.stderr,
                        )
                except Exception as e:
                    print(
                        f"Warning: Failed to parse HParamsPluginData from {file_path}: {e}",
                        file=sys.stderr,
                    )
                    return None  # Indicate parsing failure
            else:
                print(
                    f"Warning: HParams tag '{script_under_test.HPARAMS_TAG}' found in {file_path}, but plugin data is missing or not for HParams plugin.",
                    file=sys.stderr,
                )
        else:
            # This is normal if the file doesn't contain HParams with the specific tag
            pass  # print(f"Debug: No HParams tag '{script_under_test.HPARAMS_TAG}' found in summary metadata for {file_path}.")
    except Exception as e:
        print(
            f"Warning: Failed to load or process event file {file_path} for HParams extraction: {e}",
            file=sys.stderr,
        )
        return None  # Indicate loading failure

    return None  # No HParams found or other issue


def create_tfevents_file(file_dir: str, actions: list[tuple]) -> list[str]:
    """Creates tfevents file(s) using SummaryWriter based on a list of actions.
    Writes files into a subdirectory within file_dir based on SummaryWriter's default naming.
    Returns a list of paths to the generated event files.
    """
    # SummaryWriter will create a subdirectory inside file_dir
    writer = SummaryWriter(log_dir=file_dir)
    writer_log_dir = writer.log_dir  # Capture the actual subdirectory path

    hparams_added = False

    for action in actions:
        action_type = action[0]
        if action_type == "scalar":
            _, tag, value, step = action
            writer.add_scalar(tag, value, step)
        elif action_type == "hparams":
            _, hparams_dict, metrics_dict = action
            writer.add_hparams(
                hparams_dict, metrics_dict
            )  # Pass metrics_dict as required
            hparams_added = True
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    writer.flush()
    if hparams_added:
        import time

        time.sleep(0.1)
    writer.close()

    # Find the event file created *inside* the writer's log_dir (recursively)
    generated_files = list(
        glob.glob(
            os.path.join(writer_log_dir, "**", "events.out.tfevents.*"), recursive=True
        )
    )

    if not generated_files:
        raise FileNotFoundError(
            f"Could not find generated tfevents file in {writer_log_dir}"
        )

    return generated_files


@pytest.fixture
def temp_dir():
    """Pytest fixture for creating a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_copy_passthrough_no_hparams(temp_dir):
    """Case 1: Test file copy when no HParams are present."""
    base_output_name = "output_no_hparams"
    actions = [
        ("scalar", "loss", 0.5, 1),
        ("scalar", "accuracy", 0.9, 1),
        ("scalar", "loss", 0.4, 2),
    ]
    # create_tfevents_file returns list[str], no longer needs base_file_name
    input_file_paths = create_tfevents_file(temp_dir, actions)
    assert input_file_paths, "Event file creation failed or no files found"

    for idx, input_path in enumerate(input_file_paths):
        print(f"Processing file {idx + 1}/{len(input_file_paths)}: {input_path}")
        output_name = f"{base_output_name}_{idx}.tfevents"
        output_path = os.path.join(temp_dir, output_name)

        # Check redaction need for this specific file (should be False)
        needs_redact = script_under_test.check_needs_redaction(input_path)
        assert not needs_redact, f"File {input_path} unexpectedly requires redaction."

        # Run the main logic (mocking args)
        test_args = ["script_name", input_path, output_path]
        with patch.object(sys, "argv", test_args):
            script_under_test.main()

        # Verify output file exists and is identical to input
        assert os.path.exists(output_path), f"Output file {output_path} not found."
        assert filecmp.cmp(input_path, output_path, shallow=False), (
            f"File {input_path} and {output_path} differ, but no redaction was expected."
        )


def test_copy_passthrough_hparams_no_paths(temp_dir):
    """Case 2: Test file copy when HParams exist but contain no paths."""
    base_output_name = "output_hparams_no_paths"
    hparams = {
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "use_gpu": True,
        "model_name": "my_model_v1",
    }
    metrics_dict = {"hparam/metric": 0}
    actions = [
        ("scalar", "loss", 0.4, 1),
        ("hparams", hparams, metrics_dict),  # Add hparams
        ("scalar", "loss", 0.3, 2),
    ]
    # create_tfevents_file returns list[str], no longer needs base_file_name
    input_file_paths = create_tfevents_file(temp_dir, actions)
    assert input_file_paths, "Event file creation failed or no files found"

    for idx, input_path in enumerate(input_file_paths):
        print(f"Processing file {idx + 1}/{len(input_file_paths)}: {input_path}")
        output_name = f"{base_output_name}_{idx}.tfevents"
        output_path = os.path.join(temp_dir, output_name)

        # Check redaction need for this specific file (should be False)
        needs_redact = script_under_test.check_needs_redaction(input_path)
        assert not needs_redact, (
            f"File {input_path} unexpectedly requires redaction (hparams exist but no paths)."
        )

        # Run the main logic
        test_args = ["script_name", input_path, output_path]
        with patch.object(sys, "argv", test_args):
            script_under_test.main()

        # Verify output file exists and is identical to input
        assert os.path.exists(output_path), f"Output file {output_path} not found."
        assert filecmp.cmp(input_path, output_path, shallow=False), (
            f"File {input_path} and {output_path} differ, but no redaction was expected (hparams no paths)."
        )


def test_copy_and_redact_paths(temp_dir):
    """Case 3: Test file copy with redaction of HParam paths."""
    base_output_name = "output_hparams_redacted"
    hparams = {
        "config_file": "/absolute/path/to/config.yaml",  # Needs redaction
        "dataset_path": "/data/my_dataset",  # Needs redaction
        "learning_rate": 0.01,  # No redaction
        "relative_path": "relative/model.ckpt",  # No redaction
        "empty_path": "",  # No redaction
    }
    metrics_dict = {"hparam/accuracy": 0.95}
    actions = [
        ("scalar", "val_loss", 1.5, 1),
        ("hparams", hparams, metrics_dict),  # Add hparams
    ]
    # create_tfevents_file returns list[str], no longer needs base_file_name
    input_file_paths = create_tfevents_file(temp_dir, actions)
    assert input_file_paths, "Event file creation failed or no files found"

    # Keep track if we found at least one file that needed redaction
    at_least_one_file_needed_redaction = False

    for idx, input_path in enumerate(input_file_paths):
        # breakpoint()
        print(f"Processing file {idx + 1}/{len(input_file_paths)}: {input_path}")
        output_name = f"{base_output_name}_{idx}.tfevents"
        output_path = os.path.join(temp_dir, output_name)

        # Check if this specific file needs redaction
        needs_redact = script_under_test.check_needs_redaction(input_path)
        if needs_redact:
            at_least_one_file_needed_redaction = True
            print(f"  File {input_path} requires redaction.")
        else:
            print(f"  File {input_path} does not require redaction.")

        # Run the main logic
        test_args = ["script_name", input_path, output_path]
        with patch.object(sys, "argv", test_args):
            script_under_test.main()

        # Verify output file exists
        assert os.path.exists(output_path), f"Output file {output_path} not found."

        # Perform verification based on whether this specific file needed redaction
        if needs_redact:
            assert not filecmp.cmp(input_path, output_path, shallow=False), (
                f"File {input_path} and {output_path} are identical, but redaction was expected."
            )

            # --- Verification using EventAccumulator via helper ---
            # Check if output_path is a directory before trying to load it
            if os.path.isdir(output_path):
                print(
                    f"Error: Output path {output_path} is a directory before HParams extraction! Contents: {os.listdir(output_path)}",
                    file=sys.stderr,
                )
                pytest.fail(f"Output path {output_path} is a directory, not a file.")
            else:
                print(
                    f"Debug: Output path {output_path} is a file, proceeding with HParams extraction."
                )

            extracted_redacted_hparams = extract_hparams_from_event_file(output_path)
            # breakpoint()
            assert extracted_redacted_hparams is not None, (
                f"HParams could not be extracted from the redacted output file: {output_path}"
            )

            # Check redacted values
            assert "config_file" in extracted_redacted_hparams
            assert (
                extracted_redacted_hparams["config_file"]
                == script_under_test.REDACTED_VALUE
            )
            assert "dataset_path" in extracted_redacted_hparams
            assert (
                extracted_redacted_hparams["dataset_path"]
                == script_under_test.REDACTED_VALUE
            )
            # Check non-redacted values
            assert "learning_rate" in extracted_redacted_hparams
            assert extracted_redacted_hparams["learning_rate"] == pytest.approx(0.01)
            assert "relative_path" in extracted_redacted_hparams
            assert extracted_redacted_hparams["relative_path"] == "relative/model.ckpt"
            assert "empty_path" in extracted_redacted_hparams
            assert extracted_redacted_hparams["empty_path"] == ""
        else:
            # If this specific file (e.g., potentially one with only scalars if SW split them)
            # didn't need redaction, it should be identical.
            assert filecmp.cmp(input_path, output_path, shallow=False), (
                f"File {input_path} and {output_path} differ, but no redaction was needed for this specific file."
            )

    # After processing all files, assert that at least one file actually contained
    # the HParams needing redaction, otherwise the test case setup is faulty.
    assert at_least_one_file_needed_redaction, (
        "Test setup error: No input file requiring redaction was found among generated files."
    )
