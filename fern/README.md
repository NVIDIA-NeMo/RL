# NeMo RL Fern Documentation

This folder contains the Fern Docs configuration for NeMo RL.

## Installation

```bash
npm install -g fern-api
# Or: npx fern-api --version
```

## Local Preview

```bash
cd fern/
fern docs dev
# Or from project root: fern docs dev --project ./fern
```

Docs available at `http://localhost:3000`.

## Folder Structure

```
fern/
├── docs.yml              # Global config (title, colors, versions)
├── fern.config.json      # Fern CLI config
├── versions/
│   └── v0.5.0.yml       # Navigation for v0.5.0
├── v0.5.0/
│   └── pages/            # MDX content for v0.5.0
├── scripts/              # Migration and conversion scripts
└── assets/               # Favicon, images
```

## Migration Workflow

To migrate or update docs from `docs/` to Fern:

**Assets:** The docs reference images (e.g. `../assets/*.png`). These must exist in `docs/assets/` and will be copied to `fern/assets/` by the copy script. If `docs/assets/` is missing or images are not committed, create the directory and add the image files, or the Fern build will report missing path errors. Image paths in MDX are normalized to `/assets/` (relative to the Fern site root).

```bash
# 1. Copy docs to fern (run from repo root)
python3 fern/scripts/copy_docs_to_fern.py v0.5.0

# 2. Convert RL-specific syntax first (octicon, py:class, py:meth)
python3 fern/scripts/convert_rl_specific.py fern/v0.5.0/pages

# 3. Convert MyST to Fern MDX
python3 fern/scripts/convert_myst_to_fern.py fern/v0.5.0/pages

# 4. Add frontmatter
python3 fern/scripts/add_frontmatter.py fern/v0.5.0/pages

# 5. Update internal links
python3 fern/scripts/update_links.py fern/v0.5.0/pages

# 6. Remove duplicate H1s (when title matches frontmatter)
python3 fern/scripts/remove_duplicate_h1.py fern/v0.5.0/pages

# 7. Validate
./fern/scripts/check_unconverted.sh fern/v0.5.0/pages
uv run python fern/scripts/find_tag_mismatches.py fern/v0.5.0/pages
```

## Bumping the Version

When releasing a new version (e.g., v0.6.0):

1. Copy the previous version's content:
   ```bash
   cp -r fern/v0.5.0 fern/v0.6.0
   ```

2. Create the navigation file:
   ```bash
   cp fern/versions/v0.5.0.yml fern/versions/v0.6.0.yml
   ```

3. In `versions/v0.6.0.yml`: replace `../v0.5.0/pages/` → `../v0.6.0/pages/`

4. In `docs.yml`: add the new version to the `versions:` list

5. Make content changes in `fern/v0.6.0/pages/`

## MDX Components

```mdx
<Note>Informational note</Note>
<Tip>Helpful tip</Tip>
<Warning>Warning message</Warning>
<Info>Info callout</Info>

<Cards>
  <Card title="Title" href="/path">Description</Card>
</Cards>

<Tabs>
  <Tab title="Python">```python\ncode\n```</Tab>
</Tabs>
```

## API Reference

API docs are built by Sphinx (autodoc2) and hosted at docs.nvidia.com. The "API Reference" link in the navbar points to `https://docs.nvidia.com/nemo/rl/latest/apidocs/`.

## Deploying

```bash
fern generate --docs
fern docs deploy
```

## Useful Links

- [Fern Docs](https://buildwithfern.com/learn/docs)
- [MDX Components](https://buildwithfern.com/learn/docs/components)
- [Versioning Guide](https://buildwithfern.com/learn/docs/configuration/versions)
