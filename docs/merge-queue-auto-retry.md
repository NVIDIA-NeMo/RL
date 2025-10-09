# Merge Queue Auto-Retry

This repository includes an automated system to retry pull requests that are removed from the GitHub merge queue due to status check timeouts or other transient issues.

## How It Works

The system is implemented as a GitHub Action workflow (`.github/workflows/merge-queue-retry.yml`) that:

1. **Listens for dequeue events**: Triggers when a PR is removed from the merge queue (`pull_request.dequeued` event)
2. **Tracks retry attempts**: Counts previous retry attempts using PR comments to avoid infinite loops
3. **Waits before retrying**: Adds a 30-second delay to avoid immediate re-dequeue
4. **Automatically requeues**: Uses GitHub's GraphQL API to add the PR back to the merge queue
5. **Provides feedback**: Adds comments to the PR documenting retry attempts
6. **Enforces limits**: Stops retrying after 3 attempts to prevent spam

## Configuration

### Maximum Retry Attempts

The default maximum retry limit is **3 attempts**. You can modify this by editing the `MAX_RETRIES` variable in the workflow:

```yaml
# Maximum retry attempts (configurable)
MAX_RETRIES=3
```

### Required Permissions

The workflow uses the default `GITHUB_TOKEN` which should have sufficient permissions for most repositories. If you encounter permission issues, you may need to:

1. Ensure the workflow has `pull-requests: write` permissions
2. Create a Personal Access Token with appropriate scopes if needed

## Behavior

### When a PR is dequeued:

1. **First retry**: The workflow will immediately attempt to requeue the PR after a 30-second wait
2. **Subsequent retries**: Each retry is documented with a comment and counted
3. **Maximum retries reached**: After 3 failed attempts, the workflow stops and notifies that manual intervention is needed
4. **Error handling**: If the workflow itself fails, it will add an error comment to the PR

### Safety Features

- **Draft PR exclusion**: Draft PRs are not automatically retried
- **Retry counting**: Previous attempts are tracked to prevent infinite loops
- **Fallback mechanism**: If the GraphQL API fails, it falls back to using GitHub CLI
- **User notification**: All retry attempts and failures are documented in PR comments

## Troubleshooting

### Common Issues

1. **Permission errors**: Ensure the workflow has appropriate permissions to modify PRs
2. **API rate limits**: The workflow includes delays to minimize rate limiting issues
3. **Conflicting PRs**: If a PR has merge conflicts, it may continue to be dequeued

### Manual Override

If you need to disable auto-retry for a specific PR, you can:
1. Add a comment with the text "disable auto-retry" (feature not yet implemented)
2. Convert the PR to draft status temporarily
3. Manually remove the PR from the merge queue

## Monitoring

You can monitor the auto-retry system by:
- Checking PR comments for retry notifications
- Reviewing the workflow run logs in the Actions tab
- Looking for patterns in repeatedly failing PRs

## Customization

The workflow can be customized by modifying:
- Maximum retry attempts
- Wait time between retries
- Comment messages
- Conditions for when to retry (e.g., specific failure reasons)

## Benefits

- **Reduces manual intervention**: Automatically handles transient merge queue issues
- **Improves developer experience**: PRs don't get stuck due to temporary CI issues
- **Maintains safety**: Built-in limits prevent infinite retry loops
- **Provides transparency**: All actions are logged and communicated to developers




