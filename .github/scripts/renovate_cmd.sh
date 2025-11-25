#!/bin/bash
# Custom Renovate command that configures git safe.directory before running renovate
# This is needed because the pre-cloned repo is owned by a different user than the container user

# Mark all directories as safe (required for pre-cloned repos with different ownership)
git config --global --add safe.directory '*'

# Enable auto-sync of CACHED_DEPENDENCIES in 3rdparty setup.py files
# This allows submodule updates to proceed - the setup.py will use submodule deps directly
export NRL_AUTO_SYNC_DEPS=1

# Run the actual renovate command
exec renovate "$@"

