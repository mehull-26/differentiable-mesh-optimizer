# Installation Guide

This guide provides detailed instructions for installing the 3D Model Reconstruction project.

## Prerequisites

### System Requirements

- **Operating System**: Windows, Linux, or macOS
- **Python**: 3.8 or higher
- **GPU** (recommended): NVIDIA GPU with CUDA support for 160-240x speedup
- **RAM**: Minimum 8GB (16GB recommended)

### Software Dependencies

- **Python 3.8+**
- **CUDA Toolkit 11.7+** (for GPU acceleration)
- **Visual Studio Build Tools** (Windows only, for PyTorch3D compilation)

## Installation Methods

### Method 1: Conda Environment (Recommended)

#### Step 1: Create Conda Environment

```bash
# Create new environment with Python 3.11
conda create -n 3dModelRecon python=3.11 -y

# Activate environment
conda activate 3dModelRecon
```

#### Step 2: Install PyTorch with CUDA

Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) to get the correct command for your system.

```bash
# Example for CUDA 11.8 (check PyTorch website for latest)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

For CPU-only installation:

```bash
conda install pytorch torchvision cpuonly -c pytorch -y
```

#### Step 3: Clone Repository

```bash
git clone <repository-url>
cd "3d model reconstruction"
```

#### Step 4: Install Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt
```

#### Step 5: Install PyTorch3D

```bash
# Install from source (recommended)
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

**Note for Windows users**: You may need Visual Studio Build Tools. Download from [Microsoft](https://visualstudio.microsoft.com/downloads/) and install "Desktop development with C++".

### Method 2: Virtual Environment (venv)

#### Step 1: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate
```

#### Step 2: Install PyTorch

```bash
# Visit https://pytorch.org/ for correct command
pip install torch torchvision
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

## Verification

### Verify Installation

Run this command to verify all dependencies are installed correctly:

```bash
python -c "import torch; import pytorch3d; import matplotlib; import yaml; print('✓ All dependencies installed'); print(f'PyTorch: {torch.__version__}'); print(f'PyTorch3D: {pytorch3d.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Expected output:
```
✓ All dependencies installed
PyTorch: 2.0.0+cu118
PyTorch3D: 0.7.5
CUDA Available: True
```

### Run Tests

```bash
# Run unit tests
pytest tests/ -v

# Expected: 36 passed
```

### Generate Test Data

```bash
# Generate synthetic test cases
python -m data.generate_targets

# Check generated files
ls data/test_cases/test_case_a/
```

Expected output:
```
camera_config.json
ground_truth.obj
initial.obj
metadata.json
visualization.png
targets_hard/
targets_soft/
```

### Run Quick Test

```bash
# Run quick optimization (50 iterations, ~30 seconds on GPU)
python main.py --config configs/config_quick_test.yaml
```

If this completes successfully, your installation is working!

## Troubleshooting

### Common Issues

#### Issue 1: PyTorch3D Installation Fails

**Solution**: Try installing pre-built wheels:

```bash
# For Python 3.11, PyTorch 2.0, CUDA 11.8
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py311_cu118_pyt200/download.html
```

Check [PyTorch3D installation page](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for other versions.

#### Issue 2: CUDA Out of Memory

**Solution**: Reduce image size or batch processing:

In `configs/config.yaml`:
```yaml
rendering:
  image_size: 128  # Reduce from 256
```

#### Issue 3: Module Not Found Error

**Solution**: Ensure project root is in Python path:

```bash
# Add to Python path (temporary)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/macOS
$env:PYTHONPATH = "${env:PYTHONPATH};$(Get-Location)"  # Windows PowerShell
```

Or run with module syntax:
```bash
python -m data.generate_targets
```

#### Issue 4: OpenMP Duplicate Library Warning

**Solution**: Set environment variable:

```bash
# Windows
$env:KMP_DUPLICATE_LIB_OK="TRUE"

# Linux/macOS
export KMP_DUPLICATE_LIB_OK=TRUE
```

Or install nomkl:
```bash
conda install nomkl
```

### Platform-Specific Notes

#### Windows

- Install Visual Studio Build Tools before PyTorch3D
- Use PowerShell or Command Prompt (not Git Bash)
- Path might need quotes: `cd "3d model reconstruction"`

#### Linux

- Ensure CUDA drivers are up to date: `nvidia-smi`
- May need build essentials: `sudo apt-get install build-essential`

#### macOS

- PyTorch3D supports M1/M2 chips (CPU only, no Metal acceleration yet)
- Install Xcode command line tools: `xcode-select --install`

## Performance Optimization

### For GPU Users

```bash
# Check GPU usage during optimization
nvidia-smi -l 1
```

Expected GPU utilization: 60-90% during rendering

### For CPU Users

The system works on CPU but is slower (160-240x slower than GPU):
- Quick test: ~10-15 minutes
- Full optimization: ~2-3 hours

## Updating

To update to the latest version:

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Regenerate test cases if config changed
python -m data.generate_targets
```

## Uninstallation

### Conda Environment

```bash
# Deactivate environment
conda deactivate

# Remove environment
conda env remove -n 3dModelRecon
```

### Virtual Environment

```bash
# Deactivate environment
deactivate

# Remove directory
rm -rf venv  # Linux/macOS
Remove-Item -Recurse -Force venv  # Windows PowerShell
```

## Next Steps

After successful installation:

1. Read [USAGE.md](USAGE.md) for running experiments
2. Check [CONFIGURATION.md](CONFIGURATION.md) for customization options
3. Explore [API.md](API.md) for code structure

## Getting Help

If you encounter issues:

1. Check [Troubleshooting](#troubleshooting) section above
2. Search existing [GitHub Issues](../../issues)
3. Open a new issue with:
   - Error message
   - Python version (`python --version`)
   - PyTorch version
   - Operating system
   - Installation method used
