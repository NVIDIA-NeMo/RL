#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(realpath ${SCRIPT_DIR}/..)

set -eou pipefail

cd $SCRIPT_DIR

if ! command -v pytest >/dev/null 2>&1; then
    echo "[ERROR] pytest not found. Make sure it's installed."
    exit 1
elif ! command -v ray >/dev/null 2>&1; then
    echo "[ERROR] ray binary not installed, which suggests this package is not installed."
    exit 1
fi

# First try to connect to a ray cluster
if ! ray status &>/dev/null; then
    # If we cannot, then check if the local machine has at least two gpus to run the tests
    GPUS_PER_NODE=$(nvidia-smi -L | grep -c '^GPU')
    if [[ $GPUS_PER_NODE -lt 2 ]]; then
        echo "[ERROR]: Unit tests need at least 2 GPUs, but found $GPUS_PER_NODE"
        exit 1
    fi
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    nvidia-smi
    export CUDA_VISIBLE_DEVICES=0,1
fi

export PYTHONPATH=$(realpath ${SCRIPT_DIR}/..):${PYTHONPATH:-}

# Run unit tests
echo "Running unit tests..."
if [[ "$#" -eq 0 ]]; then
    pytest_args="unit/"
elif [[ "$1" != unit/* ]]; then
    pytest_args="unit/ $@"
else
    pytest_args="$@"
fi

if ! pytest $pytest_args; then
    echo "[ERROR]: Unit tests failed."
    exit 1
fi
echo "Unit tests passed!"
