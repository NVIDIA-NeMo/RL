#!/bin/bash

# This script packages all release runs into a tarball with a git SHA so that we can upload to our
# release page. The SHA is to avoid conflicts with previous runs, but when we upload we should
# remove that so that users can expect that the name is release_runs.tar.gz (this renaming can be
# done in the Github Release UI).

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/..)
cd $PROJECT_ROOT

set -eou pipefail
# Enable recursive globbing
shopt -s globstar

OUTPUT_TAR="release_runs-$(git rev-parse --short HEAD).tar.gz"

TB_EVENTS=$(ls code_snapshots/*/tests/test_suites/**/logs/*/tensorboard/**/events* || true)

# Check if the glob expanded to any files
if [[ -z "$TB_EVENTS" ]]; then
    echo "Error: No tensorboard event files found matching the pattern."
    exit 1
elif [[ -f $OUTPUT_TAR ]]; then
    echo "Error: $OUTPUT_TAR already exists. Clean it up before continuing."
    exit 1
fi

TMP_DIR=$(mktemp -d)
echo "Created temporary directory: $TMP_DIR"

# Set up trap to clean up temporary directory on exit
trap "echo 'Cleaning up temporary directory $TMP_DIR'; rm -rf $TMP_DIR" EXIT

# Loop over all the recipe runs and package them into a tarball
for tbevent in $TB_EVENTS; do
    exp_name=$(basename -- $(cut -d/ -f2 <<<$tbevent) -logs)
    # Redact the hostname
    # events.out.tfevents.1744822578.<host-name>.780899.0
    redacted_event_path=$(basename $tbevent | awk -F. '{print $1"."$2"."$3"."$4".HOSTNAME."$(NF-1)"."$NF}')
    
    # Create subdirectory for experiment if it doesn't exist
    mkdir -p "$TMP_DIR/$exp_name"

    if [[ -f "$TMP_DIR/$exp_name/$redacted_event_path" ]]; then
        echo "Error: $redacted_event_path already exists. This is unusual since tensorboard usually suffixes event files with a unique number. Please investigate."
        exit 1
    fi
    
    # Copy the event file with redacted hostname to the experiment subdirectory
    uv run --with tensorboard --no-project tools/copy_tbevent_maybe_redact.py "$tbevent" "$TMP_DIR/$exp_name/$redacted_event_path"
    
    echo "[$exp_name] Copied $tbevent to $TMP_DIR/$exp_name/$redacted_event_path"
done

# Create a tarball of all the processed event files
tar -czf "$OUTPUT_TAR" -C "$TMP_DIR" .
echo "Created tarball: $OUTPUT_TAR"
