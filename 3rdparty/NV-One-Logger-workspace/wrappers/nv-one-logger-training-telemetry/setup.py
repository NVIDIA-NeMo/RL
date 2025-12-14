import sys
import toml
from pathlib import Path
import setuptools

# Define the source directory for the actual package (git-cloned repo)
src_dir = Path(__file__).parent.parent.parent / "nv-one-logger" / "nv_one_logger" / "one_logger_training_telemetry"

if not src_dir.exists():
    raise FileNotFoundError(f"{src_dir} not found.")

# Read the underlying pyproject.toml
pyproject_toml_path = src_dir / "pyproject.toml"
with pyproject_toml_path.open("r") as f:
    pyproject_data = toml.load(f)

# Extract dependencies
# Handle poetry dependencies format
poetry_deps = pyproject_data.get("tool", {}).get("poetry", {}).get("dependencies", {})
install_requires = []
for pkg, ver in poetry_deps.items():
    if pkg == "python":
        continue
    # Simple conversion, might need more robust handling for complex version strings
    # but sufficient for this specific repo's needs
    if isinstance(ver, str):
        if ver.startswith("^"):
            install_requires.append(f"{pkg}>={ver[1:]}")
        else:
            install_requires.append(f"{pkg}=={ver}")
    elif isinstance(ver, dict):
        # Handle cases like {version = "...", markers = "..."} if present
        if "version" in ver:
            version_spec = ver['version']
            # If version already has operators (>=, ==, etc.), use as-is
            if any(op in version_spec for op in ['>=', '<=', '==', '!=', '~=', '>', '<']):
                install_requires.append(f"{pkg}{version_spec}")
            else:
                install_requires.append(f"{pkg}>={version_spec}")

# Extract version
version = pyproject_data.get("tool", {}).get("poetry", {}).get("version", "0.0.0")

# Map the package directory
# The package name in source is 'nv_one_logger' (from src directory in one_logger_training_telemetry)
# one_logger_training_telemetry/src/nv_one_logger
package_dir = {
    "": str(src_dir / "src")
}
packages = setuptools.find_packages(where=str(src_dir / "src"))

setuptools.setup(
    name="nv-one-logger-training-telemetry",
    version=version,
    description="Training job telemetry using OneLogger library.",
    packages=packages,
    package_dir=package_dir,
    install_requires=install_requires,
)
