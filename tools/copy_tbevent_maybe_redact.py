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
import argparse
import glob
import os
import shutil
import sys
import tempfile
from typing import Dict, Set  # Import necessary types

from tensorboard import errors as tb_errors
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.plugins.hparams import (
    api_pb2,  # Import the api_pb2 module
    plugin_data_pb2,
)
from tensorboard.summary.writer.event_file_writer import EventFileWriter

# By default TB tries to be smart about what to load in memory to avoid OOM
# Since we expect every step to be there when we do our comparisons, we explicitly
# set the size guidance to 0 so that we load everything. It's okay given our tests
# are small/short.
SIZE_GUIDANCE_META: Dict[str, int | bool] = {
    event_accumulator.SCALARS: 0,
    event_accumulator.IMAGES: 0,
    event_accumulator.AUDIO: 0,
    event_accumulator.HISTOGRAMS: 0,
    event_accumulator.TENSORS: 0,
    event_accumulator.GRAPH: False,
    event_accumulator.RUN_METADATA: False,
}

HPARAMS_TAG: str = "_hparams_/session_start_info"
HPARAMS_PLUGIN_NAME: str = "hparams"
REDACTED_VALUE: str = "[REDACTED]"


def check_needs_redaction(input_path: str) -> bool:
    """Checks if the TensorBoard event file contains HParams with string values starting with '/'.

    Returns True if redaction is needed, False otherwise.
    Uses EventAccumulator with minimal size guidance for speed.
    """
    try:
        print(f"Checking for HParams needing redaction in: {input_path}")
        ea = event_accumulator.EventAccumulator(
            input_path, size_guidance=SIZE_GUIDANCE_META
        )
        ea.Reload()

        if HPARAMS_TAG in ea.summary_metadata:
            metadata = ea.summary_metadata[HPARAMS_TAG]
            if (
                metadata.plugin_data
                and metadata.plugin_data.plugin_name == HPARAMS_PLUGIN_NAME
                and metadata.plugin_data.content
            ):
                try:
                    plugin_data: plugin_data_pb2.HParamsPluginData = (
                        plugin_data_pb2.HParamsPluginData.FromString(
                            metadata.plugin_data.content
                        )
                    )
                    if plugin_data.HasField("session_start_info"):
                        hparams_map: Dict[str, api_pb2.HParamValue] = (
                            plugin_data.session_start_info.hparams
                        )
                        # Check VALUES now, not keys
                        for key, hparam_value in hparams_map.items():
                            # Check if it's a string value and starts with '/'
                            if hparam_value.HasField(
                                "string_value"
                            ) and hparam_value.string_value.startswith("/"):
                                print(
                                    f"  Found HParam needing redaction: key='{key}', value='{hparam_value.string_value}'"
                                )
                                return True
                        print(
                            "  Found HParams, but no string values start with '/'. No redaction needed."
                        )
                        return False
                    else:
                        print(
                            "  HParams tag metadata found, but no session_start_info field. Assuming no redaction needed."
                        )
                        return False
                except Exception as parse_err:
                    print(
                        f"Warning: Error parsing HParams plugin data: {parse_err}. Assuming no redaction needed.",
                        file=sys.stderr,
                    )
                    return False
            else:
                print(
                    "  Found HParams tag metadata, but plugin data is missing, invalid, or empty. Assuming no redaction needed."
                )
                return False
        else:
            print(
                f"  No HParams tag ('{HPARAMS_TAG}') found in summary metadata. No redaction needed."
            )
            return False
    except (tb_errors.CorruptEventFileError, tb_errors.DataLossError, Exception) as e:
        print(
            f"Warning: Error reading or processing event file for redaction check: {e}. Assuming no redaction needed.",
            file=sys.stderr,
        )
        return False


def redact_hparams_and_write(input_path: str, output_path: str) -> None:
    """Reads events from input_path, redacts hparam string values that start with '/'.

    Writes all events to output_path using TensorBoard's native utilities.
    """
    print(f"Redacting HParams from '{input_path}' and writing to '{output_path}'")
    redacted_count: int = 0
    event_count: int = 0
    hparam_event_found: bool = False
    hparam_event_modified: bool = False

    parent_dir = os.path.dirname(output_path)
    if not parent_dir:
        parent_dir = "."
    os.makedirs(parent_dir, exist_ok=True)

    writer = None  # Initialize for the finally block
    with tempfile.TemporaryDirectory(
        dir=parent_dir, prefix=".writer_temp_"
    ) as temp_writer_dir:
        try:
            writer = EventFileWriter(temp_writer_dir)

            loader = EventFileLoader(input_path)
            for event in loader.Load():
                event_count += 1
                new_event = event_pb2.Event()
                new_event.CopyFrom(event)

                is_hparam_event: bool = False
                summary_value_index: int = -1
                if event.HasField("summary"):
                    for i, value in enumerate(event.summary.value):
                        if (
                            value.tag == HPARAMS_TAG
                            and value.metadata
                            and value.metadata.plugin_data
                            and value.metadata.plugin_data.plugin_name
                            == HPARAMS_PLUGIN_NAME
                            and value.metadata.plugin_data.content
                        ):
                            is_hparam_event = True
                            summary_value_index = i
                            hparam_event_found = True
                            break

                if is_hparam_event:
                    print(f"  Processing HParams event (Event #{event_count})...")
                    original_value: summary_pb2.Summary.Value = event.summary.value[
                        summary_value_index
                    ]
                    original_metadata: summary_pb2.SummaryMetadata = (
                        original_value.metadata
                    )
                    original_plugin_content: bytes = (
                        original_metadata.plugin_data.content
                    )

                    try:  # Keep this try-except for parsing individual HParam events
                        plugin_data_proto: plugin_data_pb2.HParamsPluginData = (
                            plugin_data_pb2.HParamsPluginData.FromString(
                                original_plugin_content
                            )
                        )
                        if plugin_data_proto.HasField("session_start_info"):
                            keys_to_redact: Set[str] = set()
                            hparams_map: Dict[str, api_pb2.HParamValue] = (
                                plugin_data_proto.session_start_info.hparams
                            )
                            for key, hparam_value_obj in hparams_map.items():
                                if hparam_value_obj.HasField(
                                    "string_value"
                                ) and hparam_value_obj.string_value.startswith("/"):
                                    keys_to_redact.add(key)

                            if keys_to_redact:
                                hparam_event_modified = True
                                print(
                                    f"    Redacting values for keys: {list(keys_to_redact)}"
                                )
                                new_plugin_data_obj: plugin_data_pb2.HParamsPluginData = plugin_data_pb2.HParamsPluginData()
                                new_plugin_data_obj.CopyFrom(plugin_data_proto)

                                for key_to_modify in keys_to_redact:
                                    if (
                                        key_to_modify
                                        in new_plugin_data_obj.session_start_info.hparams
                                    ):
                                        hparam_value_entry = new_plugin_data_obj.session_start_info.hparams[
                                            key_to_modify
                                        ]
                                        hparam_value_entry.string_value = REDACTED_VALUE
                                        redacted_count += 1
                                    else:
                                        print(
                                            f"Warning: Key '{key_to_modify}' for redaction not found in copied hparams map.",
                                            file=sys.stderr,
                                        )

                                new_plugin_data_content: bytes = (
                                    new_plugin_data_obj.SerializeToString()
                                )
                                new_summary_metadata = summary_pb2.SummaryMetadata()
                                new_summary_metadata.CopyFrom(original_metadata)
                                new_summary_metadata.plugin_data.content = (
                                    new_plugin_data_content
                                )

                                new_summary_value = summary_pb2.Summary.Value()
                                new_summary_value.CopyFrom(original_value)
                                new_summary_value.metadata.CopyFrom(
                                    new_summary_metadata
                                )
                                new_event.summary.value[summary_value_index].CopyFrom(
                                    new_summary_value
                                )
                            else:
                                print(
                                    "    HParams event found, but no values required redaction."
                                )
                        else:
                            print(
                                "    HParams event tag found, but no session_start_info field."
                            )
                    except (
                        Exception
                    ) as parse_err:  # This handles parsing of a single HParam event
                        print(
                            f"Warning: Error parsing HParams plugin data for event #{event_count}: {parse_err}. Skipping modification for this event.",
                            file=sys.stderr,
                        )
                        new_event.CopyFrom(event)

                writer.add_event(new_event)

            writer.close()  # Close writer after loop
            writer = None  # Indicate it's closed

            written_files = list(
                glob.glob(os.path.join(temp_writer_dir, "events.out.tfevents.*"))
            )
            if not written_files:
                written_files = list(glob.glob(os.path.join(temp_writer_dir, "*")))
                written_files = [
                    f
                    for f in written_files
                    if os.path.isfile(f) and ".tfevents" in f.lower()
                ]

            if len(written_files) == 1:
                writer_internal_file_path = written_files[0]
                print(f"EventFileWriter created: {writer_internal_file_path}")
                if os.path.exists(output_path):
                    if os.path.isdir(output_path):
                        print(
                            f"Removing existing directory at output path: {output_path}"
                        )
                        shutil.rmtree(output_path)
                    else:
                        print(f"Removing existing file at output path: {output_path}")
                        os.remove(output_path)
                print(f"Moving '{writer_internal_file_path}' to '{output_path}'")
                shutil.move(writer_internal_file_path, output_path)
            elif len(written_files) > 1:
                raise IOError(
                    f"Ambiguous output from EventFileWriter in {temp_writer_dir}: {written_files}"
                )
            else:
                raise IOError(
                    f"EventFileWriter failed to create an event file in {temp_writer_dir}"
                )

        except (tb_errors.CorruptEventFileError, tb_errors.DataLossError) as e:
            print(
                f"Error: Data integrity issue with input file '{input_path}'. File may be truncated or corrupted.",
                file=sys.stderr,
            )
            print(f"Specific error: {e}", file=sys.stderr)
            if os.path.exists(output_path):
                try:
                    shutil.rmtree(output_path) if os.path.isdir(
                        output_path
                    ) else os.remove(output_path)
                except OSError:
                    pass
            sys.exit(1)
        # No general except Exception here, let it propagate to main() if it's not tb_errors
        # OR, if we want to catch all from this block and clean up output_path, add it.
        # For now, assuming other errors are unexpected and should fail loudly.
        finally:
            if (
                writer is not None
            ):  # If writer was initialized but loop/move failed before explicit close
                print(
                    "Ensuring writer is closed in finally block due to an earlier error.",
                    file=sys.stderr,
                )
                writer.close()

    print(f"Finished processing {event_count} events.")
    if hparam_event_found:
        if hparam_event_modified:
            print(f"Redacted {redacted_count} HParam values.")
        else:
            print("HParams event found, but no values required redaction.")
    else:
        print(f"No HParams event ('{HPARAMS_TAG}') was found during file iteration.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy a TensorBoard event file, redacting HParam string values that start with '/'. "
        "If no HParams need redaction, performs a simple file copy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the input TensorBoard event file (tfevents*).",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to write the output (potentially redacted) event file.",
    )
    args: argparse.Namespace = parser.parse_args()

    input_f: str = args.input_path
    output_f: str = args.output_path

    if not os.path.exists(input_f):
        print(f"Error: Input file not found: '{input_f}'", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(input_f):
        print(
            f"Error: Input path must be a file, not a directory: '{input_f}'",
            file=sys.stderr,
        )
        sys.exit(1)
    if os.path.abspath(input_f) == os.path.abspath(output_f):
        print(
            f"Error: Input and output paths cannot be the same: '{input_f}'",
            file=sys.stderr,
        )
        sys.exit(1)

    needs_redact: bool = check_needs_redaction(input_f)

    if needs_redact:
        try:
            redact_hparams_and_write(input_f, output_f)
            print(f"Successfully created redacted file: '{output_f}'")
        except Exception as e:  # Generalize exception for now
            print(f"Failed to create redacted file due to error: {e}", file=sys.stderr)
            # Attempt to clean up potentially bad output file/dir
            if os.path.exists(output_f):
                try:
                    if os.path.isdir(output_f):
                        shutil.rmtree(output_f)  # Remove dir if it became one
                        print(f"Cleaned up directory: '{output_f}'")
                    else:
                        os.remove(output_f)  # Remove file
                        print(f"Cleaned up file: '{output_f}'")
                except OSError as rm_err:
                    print(
                        f"Error during cleanup of '{output_f}': {rm_err}",
                        file=sys.stderr,
                    )
            sys.exit(1)
    else:
        print(f"No redaction needed. Copying '{input_f}' to '{output_f}'...")
        try:
            output_dir: str = os.path.dirname(output_f)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            shutil.copy2(input_f, output_f)
            print(f"Successfully copied file: '{output_f}'")
        except Exception as e:
            print(f"Error during file copy: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
