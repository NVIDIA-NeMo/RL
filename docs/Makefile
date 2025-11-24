# Makefile for building documentation with isolated UV environment

.DEFAULT_GOAL := help

# Cross-platform venv paths
ifeq ($(OS),Windows_NT)
    VENV_DIR = .venv/Scripts
    PYTHON = $(VENV_DIR)/python.exe
    ACTIVATE_CMD = .venv\Scripts\activate
    RM = if exist _build rmdir /s /q _build
else
    VENV_DIR = .venv/bin
    PYTHON = $(VENV_DIR)/python
    ACTIVATE_CMD = source .venv/bin/activate
    RM = rm -rf _build
endif

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
	@echo "The environment is automatically set up on first run."
	@echo "To manually activate the docs environment, run:"
	@echo "  $(ACTIVATE_CMD)"
	@echo ""

# ------------------------------
# Ensure UV and isolated docs environment
# ------------------------------
ensure-docs-env:
	@command -v uv >/dev/null 2>&1 || ( \
		echo ""; \
		echo "❌ uv is not installed. See https://docs.astral.sh/uv/getting-started/installation/"; \
		exit 1 \
	)
	@if [ ! -x "$(PYTHON)" ]; then \
		echo "📦 Creating isolated docs environment..."; \
		uv venv .venv; \
		uv sync --project ../pyproject.toml --group docs; \
		echo "✅ Docs environment ready."; \
		echo "📝 To activate it: $(ACTIVATE_CMD)"; \
	fi

# ------------------------------
# Build HTML docs
# ------------------------------
docs-html: ensure-docs-env
	@echo "Building HTML documentation..."
	$(PYTHON) -m sphinx -b html . _build/html

# ------------------------------
# Build docs for publication (fail on warnings)
# ------------------------------
docs-publish: ensure-docs-env
	@echo "Building HTML documentation (fail on warnings)..."
	$(PYTHON) -m sphinx --fail-on-warning -b html . _build/html

# ------------------------------
# Start live-reload server
# ------------------------------
docs-live: ensure-docs-env
	@echo "Starting live-reload server..."
	$(PYTHON) -m sphinx_autobuild . _build/html --port 8001
	@echo ""
	@echo "📝 To manually activate the docs environment in a shell:"
	@echo "  $(ACTIVATE_CMD)"

# ------------------------------
# Clean built docs
# ------------------------------
docs-clean:
	@echo "Cleaning built documentation..."
	$(RM)
