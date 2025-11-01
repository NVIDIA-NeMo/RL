# Makefile for Sphinx documentation

# Default target shows help
.DEFAULT_GOAL := help

.PHONY: help docs-html docs-clean docs-live docs-publish ensure-docs-env check-uv

# Help target
help: ## Show this help message
	@echo ""
	@echo "📚 Documentation Build System"
	@echo "=============================="
	@echo ""
	@echo "Available targets:"
	@echo "  make docs-html      Build HTML documentation"
	@echo "  make docs-live      Start live-reload server"
	@echo "  make docs-publish   Build for publication (fail on warnings)"
	@echo "  make docs-clean     Clean built documentation"
	@echo ""
	@echo "Note: Environment is automatically set up on first run."
	@echo ""

# Detect OS for cross-platform compatibility
ifeq ($(OS),Windows_NT)
    VENV_PYTHON = ../.venv-docs/Scripts/python.exe
    VENV_ACTIVATE = ..\\.venv-docs\\Scripts\\activate
    VENV_ACTIVATE_PS = ..\\.venv-docs\\Scripts\\Activate.ps1
    RM_CMD = if exist _build rmdir /s /q _build
    ECHO_BLANK = @echo.
else
    VENV_PYTHON = ../.venv-docs/bin/python
    VENV_ACTIVATE = source ../.venv-docs/bin/activate
    RM_CMD = rm -rf _build
    ECHO_BLANK = @echo ""
endif

# Check if uv is installed
check-uv:
ifeq ($(OS),Windows_NT)
	@where uv >nul 2>&1 || ( \
		echo. && \
		echo ❌ uv is not installed or not in PATH && \
		echo. && \
		echo Please install uv: https://docs.astral.sh/uv/getting-started/installation/ && \
		exit 1 \
	)
else
	@command -v uv >/dev/null 2>&1 || ( \
		echo ""; \
		echo "❌ uv is not installed or not in PATH"; \
		echo ""; \
		echo "Please install uv: https://docs.astral.sh/uv/getting-started/installation/"; \
		echo ""; \
		exit 1; \
	)
endif

# Ensure docs environment exists and is up to date
ensure-docs-env: check-uv
	@if [ ! -f "$(VENV_PYTHON)" ]; then \
		echo "📦 Setting up docs environment with uv..."; \
		cd .. && uv venv .venv-docs && uv pip install --group docs --python .venv-docs; \
		echo "✅ Environment ready!"; \
	fi

docs-html: ensure-docs-env
	@echo "Building HTML documentation..."
	$(VENV_PYTHON) -m sphinx -b html . _build/html

docs-publish: ensure-docs-env
	@echo "Building HTML documentation for publication (fail on warnings)..."
	$(VENV_PYTHON) -m sphinx --fail-on-warning --builder html . _build/html

docs-clean:
	@echo "Cleaning built documentation..."
	$(RM_CMD)

docs-live: ensure-docs-env
	@echo "Starting live-reload server (sphinx-autobuild)..."
	$(VENV_PYTHON) -m sphinx_autobuild --port 8001 . _build/html
	@echo "Setting up docs virtual environment with uv..."
ifeq ($(OS),Windows_NT)
	@where uv >nul 2>&1 || ( \
		echo. && \
		echo ❌ uv is not installed or not in PATH && \
		echo. && \
		echo uv is a fast Python package installer and resolver. && \
		echo Please install it using one of the following methods: && \
		echo. && \
		echo 🪟 Windows PowerShell: && \
		echo   powershell -c "irm https://astral.sh/uv/install.ps1 | iex" && \
		echo. && \
		echo 📦 pip: && \
		echo   pip install uv && \
		echo. && \
		echo 🍺 Scoop: && \
		echo   scoop install uv && \
		echo. && \
		echo After installation, you may need to: && \
		echo 1. Restart your terminal && \
		echo 2. Or add uv to your PATH manually && \
		echo 3. Then run 'make docs-env' again && \
		echo. && \
		echo For more installation options, visit: https://docs.astral.sh/uv/getting-started/installation/ && \
		exit 1 \
	)
else
	@command -v uv >/dev/null 2>&1 || ( \
		echo ""; \
		echo "❌ uv is not installed or not in PATH"; \
		echo ""; \
		echo "uv is a fast Python package installer and resolver."; \
		echo "Please install it using one of the following methods:"; \
		echo ""; \
		echo "🐧 Linux/Unix:"; \
		echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		echo ""; \
		echo "🍺 Homebrew (macOS):"; \
		echo "  brew install uv"; \
		echo ""; \
		echo "📦 pip:"; \
		echo "  pip install uv"; \
		echo ""; \
		echo "After installation, you may need to:"; \
		echo "1. Restart your terminal or run: source ~/.bashrc (or ~/.zshrc)"; \
		echo "2. Or add uv to your PATH manually"; \
		echo "3. Then run 'make docs-env' again"; \
		echo ""; \
		echo "For more installation options, visit: https://docs.astral.sh/uv/getting-started/installation/"; \
		exit 1; \
	)
endif
	@echo "✅ uv found, creating virtual environment..."
	cd .. && uv venv .venv-docs
	cd .. && uv pip install --group docs --python .venv-docs
	$(ECHO_BLANK)
	@echo "✅ Documentation environment setup complete!"
	$(ECHO_BLANK)
	@echo "📝 Note: The environment is NOT automatically activated."
	@echo "To manually activate the docs environment, run:"
ifeq ($(OS),Windows_NT)
	@echo "  For Command Prompt: $(VENV_ACTIVATE)"
	@echo "  For PowerShell: $(VENV_ACTIVATE_PS)"
else
	@echo "  $(VENV_ACTIVATE)"
endif
	$(ECHO_BLANK)
	@echo "Once activated, you can run other docs commands like:"
	@echo "  make docs-html    # Build HTML documentation"
	@echo "  make docs-live    # Start live-reload server"
