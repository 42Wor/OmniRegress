# OmniRegress - 🚀 Administration Guide

Welcome to the **OmniRegress** admin manual! This guide covers setup, development, and advanced maintenance for your hybrid Rust/Python project.

---

## 🛠️ Development Setup

### 1. Prerequisites

- Python 3.12+
- pip
- virtualenv (recommended)
- Rust toolchain (`cargo`)
- maturin (for Rust-Python integration)

### 2. Quickstart (Arch Linux)

```bash
# Clone the repository
git clone https://github.com/yourusername/OmniRegress.git
cd OmniRegress

# Set up Python virtual environment

python3 -m venv .venv_Ubuntu
source .venv_Ubuntu/bin/activate

# ArchLinux
python3 -m venv .venv_ArchLinux
source .venv_ArchLinux/bin/activate
# windows
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
D:/OmniRegress/.venv/Scripts/Activate.ps1

# Install Python dependencies
pip install .  # Add `[dev]` if you have development extras

# Install Rust dependencies (if needed)
cargo build
```

### 3. System-wide Setup (Alternative)

```bash
sudo pacman -S python-pip rust
python -m pip install --user -e .
```
```bash
sudo apt update
sudo apt install build-essential
gcc --version
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustc --version
```
---

## 🚦 Development Workflow

### Running Tests

```bash
# Run all Python tests
pytest

# Run a specific test file
pytest omniregress/tests/test_linear.py -v

# With coverage report
pytest --cov=omniregress
```

### Building Documentation

```bash
# If using Sphinx:
pip install sphinx
sphinx-apidoc -o docs/ omniregress/
cd docs && make html
```

---

## 🧹 Maintenance Tasks

### Version Management

1. Update the version in `pyproject.toml`
2. Tag and push:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

### Dependency Management

```bash
# Add a new dependency
pip install package
pip freeze > requirements.txt

# Upgrade all dependencies
pip install --upgrade -r requirements.txt
```

---

## 🚀 Publishing

### Build & Publish to PyPI

```bash
# Build the package
pip install build
python -m build

# Upload to PyPI (requires twine)
pip install twine
twine upload dist/*
```

---

## 🦀 Rust-Python Integration

### Clean & Rebuild

```bash
cargo clean
python -m build
pip install -e .
```

or

```bash
python -m build
maturin develop
```

or

```bash
cargo build --release
pip install -e .
```
or
```bash
maturin develop --release
```
---

### Install `maturin`

- **Arch Linux:**
  ```sh
  sudo pacman -S maturin
  ```
- **pip:**
  ```sh
  pip install maturin
  ```

---

### Build & Develop Rust Extension

```sh
maturin develop --release
```

---

### Build & Publish Wheels with `maturin`

1. Build the wheel:
   ```bash
   pip install build

   python -m build
   ```
2. (Optional) Create a `wheelhouse`:
   ```bash
   mkdir -p wheelhouse
   ```
3. Copy the wheel:
   ```bash
   mkdir -Force wheelhouse
   #cp target/wheels/omniregress-*.whl wheelhouse/


   cp dist/omniregress-*.tar.gz wheelhouse/

   #.venv
   Move-Item -Path dist/omniregress-*.whl -Destination wheelhouse/

   #.venv_Ubuntu
   mv dist/omniregress-*.whl wheelhouse/


   ```
4. Upload to PyPI:
   ```bash
   twine upload wheelhouse/*
   ```

