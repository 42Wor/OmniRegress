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
```

```bash
# Upload to PyPI (requires twine)
pip install twine
twine upload dist/*### Publishing to PyPI
````

Then, clean your build artifacts just in case:

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
 maturin develop --release
```





### Building and Publishing Wheels with `maturin`

1. Build the wheel with maturin:
    ```bash
    maturin build --release
    ```

2. (Optional) Create a `wheelhouse` directory if it does not exist:
    ```bash
    mkdir -p wheelhouse
    ```

3. Copy the generated wheel to `wheelhouse` (adjust the filename as needed):
    ```bash
    cp target/wheels/omniregress-*.whl wheelhouse/
    ```

4. Upload the wheel to PyPI:
    ```bash
    twine upload wheelhouse/*
    ```