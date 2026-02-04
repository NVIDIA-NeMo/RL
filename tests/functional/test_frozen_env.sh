#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

echo "=========================================="
echo "Frozen Environment Functional Test"
echo "=========================================="
echo

# ============================================================================
# Test 1: Ray Import Across All Python Executables
# ============================================================================
echo "Test 1: Verifying ray imports across all python-* executables"
echo "---------------------------------------------------------------"

# Find all python-* executables in /usr/local/bin
PYTHON_EXECUTABLES=($(ls -1 /usr/local/bin/python-* 2>/dev/null || true))

if [ ${#PYTHON_EXECUTABLES[@]} -eq 0 ]; then
    echo "ERROR: No python-* executables found in /usr/local/bin"
    echo "This test requires frozen environment setup (NRL_CONTAINER=1)"
    exit 1
fi

echo "Found ${#PYTHON_EXECUTABLES[@]} python-* executables to test"
echo

for py_exec in "${PYTHON_EXECUTABLES[@]}"; do
    py_name=$(basename "$py_exec")
    echo -n "  Testing $py_name ... "
    
    if $py_exec -c "import ray" 2>/dev/null; then
        echo "✓ OK"
    else
        echo "✗ FAILED"
        echo "ERROR: $py_name cannot import ray"
        exit 1
    fi
done

echo
echo "Test 1: PASSED - All python-* executables can import ray"
echo

# ============================================================================
# Test 2: Mutation Detection in Frozen Environment
# ============================================================================
echo "Test 2: Verifying mutation detection in frozen environment"
echo "-----------------------------------------------------------"

# Create temporary directory for testing
TEMP_DIR=$(mktemp -d -t nemo-rl-frozen-env-test-XXXXXX)
echo "Created temporary directory: $TEMP_DIR"

# Setup cleanup trap
cleanup() {
    echo "Cleaning up temporary directory: $TEMP_DIR"
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Copy codebase using rsync with git ls-files (like code_snapshot.sh)
echo "Copying codebase to temporary directory..."
cd "$PROJECT_ROOT"
# Add --git-dir and --work-tree to ensure we can run git without the safe.directory check
rsync -a --files-from=<(
    git --git-dir="$PROJECT_ROOT/.git" --work-tree="$PROJECT_ROOT" ls-files --recurse-submodules --cached --full-name
) ./ "$TEMP_DIR/"

# Also copy .git directories so fingerprint check can determine submodule commits
echo "Copying .git metadata for fingerprint verification..."
rsync -a .git "$TEMP_DIR/"
rsync -a 3rdparty/ "$TEMP_DIR/3rdparty/" --include='*/.git' --include='*/.git/**' --exclude='*' 2>/dev/null || true

# Navigate to temp directory and set PYTHONPATH
cd "$TEMP_DIR"
export PYTHONPATH="$TEMP_DIR:${PYTHONPATH:-}"

echo
echo "Test 2a: Baseline - import nemo_rl should succeed without mutations"
echo -n "  Testing python -c 'import nemo_rl' ... "
if python -c "import nemo_rl" 2>/dev/null; then
    echo "✓ OK"
else
    echo "✗ FAILED"
    echo "ERROR: Baseline import of nemo_rl failed (should succeed)"
    exit 1
fi

echo
echo "Test 2b: Pyproject mutation - import nemo_rl should warn after mutation"
echo "  Adding newline to top of pyproject.toml..."
# Add a newline to the top of pyproject.toml
echo "" | cat - pyproject.toml > pyproject.toml.tmp && mv pyproject.toml.tmp pyproject.toml

echo -n "  Testing python -c 'import nemo_rl' (should print warning) ... "
IMPORT_OUTPUT=$(python -c "import nemo_rl" 2>&1 || true)
if echo "$IMPORT_OUTPUT" | grep -q "WARNING: Container/Code Version Mismatch Detected"; then
    echo "✓ OK (warning printed as expected)"
else
    echo "✗ FAILED"
    echo "ERROR: import nemo_rl did not print version mismatch warning after pyproject.toml mutation"
    echo "Output was:"
    echo "$IMPORT_OUTPUT"
    exit 1
fi

# Restore pyproject.toml for next test
echo "  Restoring pyproject.toml..."
cd "$PROJECT_ROOT"
rsync -a pyproject.toml "$TEMP_DIR/pyproject.toml"
cd "$TEMP_DIR"

echo
echo "Test 2c: Submodule mutation - import nemo_rl should warn after updating submodule"
echo "  Updating 3rdparty/megatron-lm submodule to HEAD of main branch..."

# Check if megatron-lm submodule exists
if [ ! -d "3rdparty/megatron-lm/.git" ]; then
    echo "  WARNING: megatron-lm submodule not initialized, skipping submodule mutation test"
else
    cd "$TEMP_DIR/3rdparty/megatron-lm"
    
    # Fetch latest from remote and checkout main
    if git fetch origin main 2>/dev/null && git checkout origin/main 2>/dev/null; then
        echo "  Successfully updated submodule to latest main"
        
        cd "$TEMP_DIR"
        echo -n "  Testing python -c 'import nemo_rl' (should print warning) ... "
        IMPORT_OUTPUT=$(python -c "import nemo_rl" 2>&1 || true)
        if echo "$IMPORT_OUTPUT" | grep -q "WARNING: Container/Code Version Mismatch Detected"; then
            echo "✓ OK (warning printed as expected)"
        else
            echo "✗ FAILED"
            echo "ERROR: import nemo_rl did not print version mismatch warning after submodule mutation"
            echo "Output was:"
            echo "$IMPORT_OUTPUT"
            exit 1
        fi
    else
        echo "  WARNING: Could not update submodule (network issue?), skipping this test"
    fi
fi

echo
echo "Test 2: PASSED - Mutation detection working correctly"
echo

# ============================================================================
# Test 3: Import Isolation Between Worker Environments
# ============================================================================
echo "Test 3: Verifying import isolation between worker environments"
echo "---------------------------------------------------------------"

# Return to project root for test 3
cd "$PROJECT_ROOT"
unset PYTHONPATH

echo
echo "Test 3a: python-MegatronPolicyWorker should have megatron.core but not nemo_automodel"

echo -n "  Testing python-MegatronPolicyWorker can import megatron.core ... "
if python-MegatronPolicyWorker -c "import megatron.core" 2>/dev/null; then
    echo "✓ OK"
else
    echo "✗ FAILED"
    echo "ERROR: python-MegatronPolicyWorker cannot import megatron.core"
    exit 1
fi

echo -n "  Testing python-MegatronPolicyWorker cannot import nemo_automodel ... "
if python-MegatronPolicyWorker -c "import nemo_automodel" 2>/dev/null; then
    echo "✗ FAILED"
    echo "ERROR: python-MegatronPolicyWorker can import nemo_automodel (should fail)"
    exit 1
else
    echo "✓ OK (import failed as expected)"
fi

echo
echo "Test 3b: python-DTensorPolicyWorkerV2 should have nemo_automodel but not megatron.core"

echo -n "  Testing python-DTensorPolicyWorkerV2 can import nemo_automodel ... "
if python-DTensorPolicyWorkerV2 -c "import nemo_automodel" 2>/dev/null; then
    echo "✓ OK"
else
    echo "✗ FAILED"
    echo "ERROR: python-DTensorPolicyWorkerV2 cannot import nemo_automodel"
    exit 1
fi

echo -n "  Testing python-DTensorPolicyWorkerV2 cannot import megatron.core ... "
if python-DTensorPolicyWorkerV2 -c "import megatron.core" 2>/dev/null; then
    echo "✗ FAILED"
    echo "ERROR: python-DTensorPolicyWorkerV2 can import megatron.core (should fail)"
    exit 1
else
    echo "✓ OK (import failed as expected)"
fi

echo
echo "Test 3: PASSED - Import isolation working correctly"
echo

# ============================================================================
# Summary
# ============================================================================
echo "=========================================="
echo "All Frozen Environment Tests PASSED ✓"
echo "=========================================="
echo
echo "Summary:"
echo "  ✓ Test 1: All ${#PYTHON_EXECUTABLES[@]} python-* executables can import ray"
echo "  ✓ Test 2: Mutation detection working (pyproject.toml and submodule changes trigger warnings)"
echo "  ✓ Test 3: Import isolation between worker environments verified"
echo

