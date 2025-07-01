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
python -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install .  # Add `[dev]` if you have development extras

# Install Rust dependencies (if needed)
cargo build
```

### 3. System-wide Setup (Alternative)

```bash
sudo pacman -S python-pip python-venv rust
python -m pip install --user -e .
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
   maturin build --release
   ```
2. (Optional) Create a `wheelhouse`:
   ```bash
   mkdir -p wheelhouse
   ```
3. Copy the wheel:
   ```bash
   cp target/wheels/omniregress-*.whl wheelhouse/
   ```
4. Upload to PyPI:
   ```bash
   twine upload wheelhouse/*
   ```

---

## 🐳 Building Manylinux Wheels with Docker

1. **Install Docker:**
   ```bash
   sudo pacman -S docker
   sudo systemctl start docker
   sudo systemctl enable docker
   sudo usermod -aG docker $USER
   newgrp docker
   ```
2. **Build inside manylinux container:**
```bash
sudo docker run --rm -v $(pwd):/io -w /io quay.io/pypa/manylinux2014_x86_64 /bin/bash -c "yum install -y gcc && curl https://sh.rustup.rs -sSf | sh -s -- -y && export PATH=/root/.cargo/bin:$PATH && /opt/python/cp312-cp312/bin/python -m pip install maturin && /opt/python/cp312-cp312/bin/maturin build --release --out dist -i /opt/python/cp312-cp312/bin/python"
```
sudo systemctl stop docker
---

✨ **Happy hacking!** ✨