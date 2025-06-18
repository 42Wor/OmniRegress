Here's a comprehensive `ADMIN.md` file for your `OmniRegress` project that documents setup, development, and maintenance procedures:


# OmniRegress - Administration Guide


## Development Setup

### 1. Prerequisites
- Python 3.12
- pip
- virtualenv (recommended)

### 2. Initial Setup (Arch Linux)

```bash
# Clone repository
git clone https://github.com/yourusername/OmniRegress.git
cd OmniRegress

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .[dev]  # If you have development extras
```

### 3. Alternative Setup (System-wide)
```bash
sudo pacman -S python-pip python-venv
python -m pip install --user -e .
```

## Development Workflow

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
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

## Maintenance Tasks

### Version Management
1. Update version in `pyproject.toml`
2. Create git tag:
```bash
git tag v0.1.0
git push origin v0.1.0
```

### Dependency Management
```bash
# Add new dependency
pip install package
pip freeze > requirements.txt

# Update all dependencies
pip install --upgrade -r requirements.txt
```
### Publishing to PyPI
```bash
# Build the package
pip install build
python -m build

# Upload to PyPI (requires twine)
pip install twine
twine upload dist/*### Publishing to PyPI


Then, clean your build artifacts just in case:W
```bash
cargo clean
```
And try building again:
```bash
pip install -e .
```
or
```bash
maturin develop
```

### Install `maturin`
You can install `maturin` using one of the following methods:

- Using `pacman` (Arch Linux):
    ```sh
    sudo pacman -S maturin
    ```

- Using `pip` (Python package manager):
    ```sh
    pip install maturin
    ```

### Build and Develop Your Rust Extension
Run the following command to build your Rust extension and make it available for Python:
```sh
maturin develop
```
Since you're on a system that doesn't use `apt-get`, let's install the dependencies using your system's package manager. Based on the error message, you might be using either Fedora, RHEL, or another RPM-based distribution.

For Fedora/RHEL-based systems:
```bash
sudo dnf install cmake gcc-gfortran openblas-devel
```

For Arch Linux:
```bash
sudo pacman -S cmake gcc-fortran openblas
sudo pacman -S openblas cblas lapack
```

After installing the dependencies, try building again:
```bash
export OPENBLAS_INCLUDE_DIR=/usr/include
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH
cargo clean
maturin develop
```

If you're using a different Linux distribution, let me know and I can provide the correct package installation command.