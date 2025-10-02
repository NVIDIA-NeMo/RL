# make docs-live-fast

## Quick Start
```bash
make docs-live-fast
```

## What It Does
- Starts a live-reload documentation server at `http://localhost:8001`
- **Skips API documentation generation** (autodoc2 disabled)
- Uses parallel builds (`-j auto`) for faster compilation
- Ignores changes to `apidocs/*` files

## When to Use

**Use `docs-live-fast` when:**
- Editing markdown guides, tutorials, or architecture docs
- Working on narrative documentation
- You want fast rebuilds (1-2 seconds vs 5-10 seconds)

**Use `docs-live` when:**
- Editing Python docstrings in `nemo_rl/`
- Updating API documentation
- You need to see API docs changes

## Speed Comparison

| Command | Initial Build | Rebuild on Save |
|---------|---------------|-----------------|
| `make docs-live-fast` | 5-10 seconds | 1-2 seconds ⚡ |
| `make docs-live` | 30-60 seconds | 5-10 seconds |

## How It Works

1. Sets `SKIP_AUTODOC=1` environment variable
2. `docs/conf.py` checks this variable and excludes `autodoc2` extension
3. Existing API documentation files still render (just won't regenerate)
4. Sphinx builds much faster without scanning Python source code

## Implementation Details

**Makefile target:**
```makefile
docs-live-fast:
	@echo "Starting fast live-reload server (skipping API docs generation)..."
	cd docs && set SKIP_AUTODOC=1 && $(DOCS_PYTHON_IN_DOCS) -m sphinx_autobuild $(if $(DOCS_ENV),-t $(DOCS_ENV)) --ignore "apidocs/*" -j auto --port 8001 . _build/html
```

**conf.py logic:**
```python
# Only include autodoc2 if not in fast mode
if not os.getenv("SKIP_AUTODOC"):
    extensions.insert(1, "autodoc2")  # Generates API docs
```

## Notes

- API documentation pages will still be visible in the browser (using existing generated files)
- Changes to Python docstrings won't update until you run `make docs-live` or `make docs-html`
- The `--ignore "apidocs/*"` flag prevents unnecessary rebuilds when API files are touched
- The `-j auto` flag enables parallel building using all available CPU cores

## Troubleshooting

**If the command doesn't work:**
1. Make sure you've run `make docs-env` to set up the documentation environment
2. Verify that `.venv-docs` exists
3. Try running `make docs-live` first to ensure everything is set up correctly

**If API docs are missing:**
- Run `make docs-live` or `make docs-html` at least once to generate the API documentation files
- The `docs/apidocs/` directory must exist with generated markdown files

