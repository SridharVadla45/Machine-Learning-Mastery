# 🛠️ Environment Setup Guide

Complete setup instructions for the Machine Learning Mastery system.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installing uv](#installing-uv)
3. [Setting Up the Project](#setting-up-the-project)
4. [Verifying Installation](#verifying-installation)
5. [IDE Setup](#ide-setup)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

Before starting, ensure you have:

1. **Python 3.10 or higher**
   ```bash
   python3 --version
   # Should show: Python 3.10.x or higher
   ```

2. **Git** (for version control)
   ```bash
   git --version
   ```

3. **Minimum 5GB free disk space** (for dependencies and datasets)

4. **Internet connection** (for downloading packages)

### Recommended Software

- **VS Code** - Best IDE for Python development
- **iTerm2** (macOS) or **Windows Terminal** - Better terminal experience
- **CUDA** (optional) - For GPU acceleration with PyTorch

---

## Installing uv

**uv** is a modern, fast Python package manager written in Rust.

### macOS / Linux

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (if not automatic)
export PATH="$HOME/.cargo/bin:$PATH"

# Verify installation
uv --version
```

### Windows

```powershell
# Using PowerShell
irm https://astral.sh/uv/install.ps1 | iex

# Verify installation
uv --version
```

### Alternative: Install via pip

```bash
pip install uv
```

---

## Setting Up the Project

### Step 1: Navigate to Project Directory

```bash
cd /path/to/machine-learning-mastery
```

### Step 2: Initialize uv Environment

```bash
# This creates a virtual environment and installs all dependencies
uv sync
```

This command will:
- Create a `.venv` virtual environment
- Install all packages from `pyproject.toml`
- Set up development dependencies

**Expected output:**
```
Creating virtualenv at .venv
Resolved XX packages in X.XXs
Installed XX packages in X.XXs
```

### Step 3: Activate Virtual Environment (Optional)

uv automatically manages the environment, but you can activate manually:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

When activated, your prompt should show: `(.venv)`

---

## Verifying Installation

### Test 1: Check Python

```bash
uv run python --version
# Should show: Python 3.10.x or higher
```

### Test 2: Check NumPy Installation

```bash
uv run python -c "import numpy as np; print(f'NumPy {np.__version__} installed!')"
```

### Test 3: Check PyTorch Installation

```bash
uv run python -c "import torch; print(f'PyTorch {torch.__version__} installed!'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Test 4: Run Example Module

```bash
uv run 00-python-fundamentals/examples.py
```

If you see output without errors, **setup is complete! 🎉**

---

## Running Modules

### Basic Usage

```bash
# Run examples (learn concepts)
uv run 01-linear-algebra/examples.py

# Run exercises (practice)
uv run 07-supervised-learning/exercises.py

# Check solutions
uv run 07-supervised-learning/solutions.py
```

### Python REPL

```bash
# Start interactive Python with all packages available
uv run python

# Or ipython for better experience
uv run ipython
```

### Jupyter Notebooks (Optional)

```bash
# Start Jupyter
uv run jupyter notebook

# Or JupyterLab
uv run jupyter lab
```

---

## IDE Setup

### VS Code (Recommended)

1. **Install VS Code**
   - Download from: https://code.visualstudio.com/

2. **Install Python Extension**
   - Open VS Code
   - Go to Extensions (⌘+Shift+X / Ctrl+Shift+X)
   - Search "Python" by Microsoft
   - Click Install

3. **Select Python Interpreter**
   ```
   ⌘+Shift+P (Ctrl+Shift+P on Windows)
   → "Python: Select Interpreter"
   → Choose: .venv/bin/python
   ```

4. **Recommended Extensions**
   - Python (Microsoft)
   - Pylance (Microsoft)
   - Jupyter (Microsoft)
   - autoDocstring
   - Black Formatter

5. **Settings (Optional)**

   Create `.vscode/settings.json`:
   ```json
   {
     "python.defaultInterpreterPath": ".venv/bin/python",
     "python.formatting.provider": "black",
     "editor.formatOnSave": true,
     "python.linting.enabled": true,
     "python.linting.pylintEnabled": false,
     "python.linting.ruff": true
   }
   ```

---

## Package Management

### Adding New Packages

```bash
# Add a package
uv add package-name

# Add a development package
uv add --dev package-name

# Example: Add scikit-image
uv add scikit-image
```

### Removing Packages

```bash
uv remove package-name
```

### Updating Packages

```bash
# Update all packages
uv sync --upgrade

# Update specific package
uv add package-name --upgrade
```

### List Installed Packages

```bash
uv pip list
```

---

## Troubleshooting

### Issue: `uv: command not found`

**Solution:**
```bash
# Add to PATH in ~/.bashrc or ~/.zshrc
export PATH="$HOME/.cargo/bin:$PATH"

# Reload shell
source ~/.bashrc  # or source ~/.zshrc
```

### Issue: Python version too old

**Solution:**
```bash
# Install Python 3.10+ using pyenv
curl https://pyenv.run | bash

# Install Python 3.11
pyenv install 3.11.0
pyenv global 3.11.0

# Verify
python --version
```

### Issue: PyTorch CUDA not available

**Solution:**

For GPU support, install PyTorch with CUDA:

```bash
# Uninstall CPU version
uv remove torch torchvision

# Install CUDA version (adjust for your CUDA version)
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Check CUDA version:
```bash
nvidia-smi
```

### Issue: Permission denied errors

**Solution:**
```bash
# Don't use sudo with uv
# If you need permissions, fix ownership:
sudo chown -R $USER:$USER ~/.cargo
```

### Issue: Module not found errors

**Solution:**
```bash
# Reinstall dependencies
rm -rf .venv
uv sync
```

### Issue: Out of memory during installation

**Solution:**
```bash
# Install packages one at a time
uv add numpy
uv add pandas
# ... etc
```

---

## GPU Setup (Optional)

### For NVIDIA GPUs

1. **Install CUDA Toolkit**
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Install version 11.8 or 12.1

2. **Verify CUDA**
   ```bash
   nvidia-smi
   nvcc --version
   ```

3. **Install PyTorch with CUDA**
   ```bash
   uv remove torch torchvision
   uv add torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Test GPU**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA device: {torch.cuda.get_device_name(0)}")
   ```

### For macOS (Apple Silicon)

PyTorch supports Metal Performance Shaders (MPS):

```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
```

---

## Data Storage

### Dataset Location

All datasets go in:
```
common/datasets/
```

### Recommended Setup

```bash
# Create dataset directories
mkdir -p common/datasets/{raw,processed,external}

# Download initial datasets (examples will do this automatically)
```

### Disk Space Requirements

- **Minimal**: ~2GB (just packages)
- **Full Course**: ~10GB (with all datasets)
- **With Projects**: ~20GB (includes trained models)

---

## Performance Optimization

### Use Multiple Cores

```python
# In your code
import os
os.environ['OMP_NUM_THREADS'] = '4'  # Adjust based on your CPU
```

### Cache Dataset Downloads

```bash
# Set HuggingFace cache location
export HF_HOME="$HOME/.cache/huggingface"

# Set torch hub cache
export TORCH_HOME="$HOME/.cache/torch"
```

### Speed Up NumPy/Pandas

```bash
# Install optimized BLAS libraries (macOS)
brew install openblas

# Linux
sudo apt-get install libopenblas-dev
```

---

## Updating the System

### Pull Latest Changes

```bash
git pull origin main
```

### Update Dependencies

```bash
uv sync --upgrade
```

---

## Alternative: Without uv

If you prefer traditional pip/venv:

```bash
# Create virtual environment
python -m venv .venv

# Activate
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install numpy pandas matplotlib scipy scikit-learn torch torchvision rich jupyter

# Run modules
python 00-python-fundamentals/examples.py
```

---

## Next Steps

✅ Environment setup complete!

**Now:**
1. Read the main `README.md`
2. Start with `00-python-fundamentals/README.md`
3. Run your first example: `uv run 00-python-fundamentals/examples.py`

**Happy Learning! 🚀**

---

## Getting Help

- **Check solutions**: Every module has detailed `solutions.py`
- **Review theory**: Read `theory.md` for conceptual help
- **Test environment**: Run verification tests above
- **Community**: Join ML learning communities

---

**Environment Version**: 1.0.0  
**Last Updated**: December 2025
