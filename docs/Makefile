# Makefile for building documentation with UV

.DEFAULT_GOAL := help

# UV run command with docs dependencies
UV_RUN = uv run --project ../pyproject.toml --group docs

# ------------------------------
# Help
# ------------------------------
help:
	@echo ""
	@echo "📚 Documentation Build System"
	@echo "=============================="
	@echo ""
	@echo "Available targets:"
	@echo "  make docs-html      Build HTML documentation"
	@echo "  make docs-live      Start live-reload server"
	@echo "  make docs-publish   Build docs (fail on warnings)"
	@echo "  make docs-clean     Clean built documentation"
	@echo ""

# ------------------------------
# Check UV is installed
# ------------------------------
check-uv:
	@command -v uv >/dev/null 2>&1 || ( \
		echo ""; \
		echo "❌ uv is not installed. See https://docs.astral.sh/uv/getting-started/installation/"; \
		exit 1 \
	)

# ------------------------------
# Build HTML docs
# ------------------------------
docs-html: check-uv
	@echo "Building HTML documentation..."
	$(UV_RUN) sphinx-build -b html . _build/html

# ------------------------------
# Build docs for publication (fail on warnings)
# ------------------------------
docs-publish: check-uv
	@echo "Building HTML documentation (fail on warnings)..."
	$(UV_RUN) sphinx-build --fail-on-warning -b html . _build/html

# ------------------------------
# Start live-reload server
# ------------------------------
docs-live: check-uv
	@echo "Starting live-reload server..."
	$(UV_RUN) sphinx-autobuild . _build/html --port 8001

# ------------------------------
# Clean built docs
# ------------------------------
docs-clean:
	@echo "Cleaning built documentation..."
	rm -rf _build
