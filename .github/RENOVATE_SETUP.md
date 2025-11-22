# Renovate Setup Documentation

This repository uses [Renovate](https://docs.renovatebot.com/) to automatically update dependencies, including git submodules and Python packages managed in `pyproject.toml`.

## What Renovate Does

Renovate automatically:
1. **Updates git submodules** by tracking the configured branches
2. **Updates a small allowlist of Python dependencies** in `pyproject.toml`:
   - `vllm`, `torch`, and `ray` for the core training stack
   - `transformer-engine` and `flash-attn` for xformers compatibility
   - `transformers` so we can track upstream releases
   - _Everything else is frozen unless explicitly requested._
3. **Syncs `3rdparty/*/setup.py` files** with their corresponding submodule dependencies
4. **Regenerates `uv.lock`** after dependency updates
5. **Creates a single PR** that automatically triggers the full CI pipeline (`cicd-main.yml`)

## Setup Requirements

You need to set up authentication for Renovate. Choose one of the following options:

### Option 1: Personal Access Token (PAT) - Quick Start

**This is the easiest way to get started:**

1. Create a GitHub Personal Access Token (PAT):
   - Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Click "Generate new token (classic)"
   - Give it a descriptive name (e.g., "Renovate Bot")
   - Select scopes:
     - ✅ `repo` (Full control of private repositories)
     - ✅ `workflow` (Update GitHub Action workflows - required for github-actions manager)
   - Click "Generate token" and copy it

2. Add the token as a repository secret:
   - Go to your repository → Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `RENOVATE_TOKEN`
   - Value: Paste your PAT
   - Click "Add secret"

3. You're done! The workflow will use the PAT automatically.

### Option 2: GitHub App (Recommended for Organizations)

**Better for rate limits and security, but requires more setup:**

1. Create a GitHub App:
   - Go to Organization Settings → Developer settings → GitHub Apps → New GitHub App
   - Or use an existing Renovate GitHub App
   
2. Configure the app with these permissions:
   - Repository permissions:
     - Contents: Read & Write
     - Pull requests: Read & Write
     - Workflows: Read & Write (if using github-actions manager)
     - Metadata: Read-only
   
3. Install the app on your repository

4. Add these secrets to your repository:
   - `RENOVATE_APP_ID`: The app ID (found on the app's settings page)
   - `RENOVATE_APP_PRIVATE_KEY`: The app's private key (PEM format)

5. The workflow will automatically detect and use the GitHub App token

### 2. Grant Workflow Permissions

Ensure the Renovate workflow has permission to:
- Create and update pull requests
- Read and write to the repository
- Access secrets

This can be configured in: `Settings` → `Actions` → `General` → `Workflow permissions`

## Configuration Files

### `.github/renovate.json`
Main configuration file that defines:
- Update schedule (daily during business hours PST)
- Package grouping rules
- Branch naming conventions
- PR labels (`dependencies`, `CI:L2`)

### `.github/workflows/renovate.yml`
GitHub Actions workflow that:
- Runs daily at 9 AM UTC (1 AM PST / 2 AM PDT)
- Can be manually triggered with `workflow_dispatch`
- Sets up the environment (Python, uv)
- Executes Renovate with proper credentials

### `.github/scripts/sync_submodule_dependencies.py`
Python script that:
- Reads dependencies from `3rdparty/*/pyproject.toml` files in submodules
- Updates `CACHED_DEPENDENCIES` in corresponding `setup.py` files
- Ensures consistency between submodule requirements and wrapper packages

### `.github/scripts/renovate_post_update.sh`
Bash script that runs after Renovate updates dependencies:
1. Syncs submodule dependencies to setup.py files
2. Runs `uv lock` to regenerate the lock file
3. Stages changes for commit

## Manual Workflow Trigger

You can manually trigger Renovate at any time:

1. Go to `Actions` → `Renovate` in GitHub
2. Click `Run workflow`
3. Optional parameters:
   - **Log level**: Set to `debug` for verbose output
   - **Dry run**: Enable to preview changes without creating PRs

## Update Strategy

Renovate now produces **one consolidated PR at a time**:

| Branch prefix | Contents | Notes |
|---------------|----------|-------|
| `renovate/allowlist-…` | Git submodules, Docker/GitHub Action updates, and the allowlisted Python packages above | Runs on the configured weekday schedule; no other dependencies are touched until explicitly re-enabled. |

## CI Integration

When Renovate creates a PR:
1. The PR is automatically labeled with `CI:L2` to trigger full CI testing
2. `cicd-main.yml` runs the complete test suite
3. All L2 tests must pass before the PR can be merged
4. The lock file and setup.py changes are included in the PR

## Troubleshooting

### Renovate workflow fails
- Check that secrets `RENOVATE_APP_ID` and `RENOVATE_APP_PRIVATE_KEY` are set
- Verify the GitHub App is installed on the repository
- Check workflow logs for specific error messages

### Dependencies not syncing
- Ensure submodules are properly initialized
- Check `.github/scripts/sync_submodule_dependencies.py` logs
- Verify that submodule `pyproject.toml` files exist and are valid

### uv lock fails
- Ensure `uv` version in workflow matches project requirements
- Check for dependency conflicts in the update
- Review the post-update script logs

### PRs not triggering CI
- Verify PR has the `CI:L2` label
- Check `cicd-main.yml` configuration
- Ensure PR is targeting the `main` branch

## Customization

To modify Renovate behavior:
1. Edit `.github/renovate.json` for scheduling, grouping, or update rules
2. Update `.github/workflows/renovate.yml` for workflow settings
3. Modify `.github/scripts/renovate_post_update.sh` for custom post-update logic

## Testing Changes

Before committing Renovate config changes:
1. Use the workflow's dry-run mode to test
2. Check the Renovate logs for validation errors
3. Test the post-update script locally:
   ```bash
   .github/scripts/renovate_post_update.sh
   ```

## References

- [Renovate Documentation](https://docs.renovatebot.com/)
- [Renovate Configuration Options](https://docs.renovatebot.com/configuration-options/)
- [GitHub Action for Renovate](https://github.com/renovatebot/github-action)

