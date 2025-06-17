Here's a comprehensive `ADMIN.md` file for your `OmniRegress` project that documents setup, development, and maintenance procedures:


# OmniRegress - Administration Guide

## Project Structure

```

OmniRegress/
├── pyproject.toml        # Build configuration
├── README.md            # Project overview
├── my_regression/       # Core package
│   ├── __init__.py
│   ├── linear_regression.py
│   ├── polynomial_regression.py
│   └── tests/
│       ├── __init__.py
│       ├── test_linear.py
│       └── test_polynomial.py
└── .venv/               # Virtual environment (ignored in git)

```

## Development Setup

### 1. Prerequisites
- Python 3.8+
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
pytest my_regression/tests/test_linear.py -v

# With coverage report
pytest --cov=my_regression
```

### Building Documentation
```bash
# If using Sphinx:
pip install sphinx
sphinx-apidoc -o docs/ my_regression/
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

