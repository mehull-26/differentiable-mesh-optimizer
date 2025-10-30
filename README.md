# 3D Model Reconstruction via Inverse Rendering

A complete PyTorch implementation of 3D shape reconstruction from 2D silhouettes using differentiable rendering and gradient-based optimization.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project implements **inverse rendering** - recovering 3D geometry from 2D observations. Given multiple silhouette images of an object from different viewpoints, the system optimizes a 3D mesh to match those observations using gradient descent.

### Key Features

- **Differentiable Rendering**: GPU-accelerated rendering with gradient flow using PyTorch3D
- **Dual Rendering Modes**: Soft rendering (smooth gradients) and hard rendering (binary silhouettes)
- **Adaptive Regularization**: Automatically scales smoothness constraints based on mesh complexity
- **Multi-View Optimization**: Reconstruct 3D shapes from multiple 2D viewpoints
- **Comprehensive Evaluation**: Chamfer distance, normal consistency, and 3D vertex matching
- **GPU Acceleration**: 160-240x speedup with CUDA support

### Example Results

The system can reconstruct various shapes from sphere initialization:

- **Ellipsoid** (Easy): Uniform axis-aligned stretching
- **Bump** (Medium): Local surface deformation
- **Cube** (Hard): Sharp edges and flat faces from smooth sphere

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd "3d model reconstruction"

# Install dependencies
pip install -r requirements.txt
```

For detailed installation instructions, see [INSTALLATION.md](docs/INSTALLATION.md).

### Generate Test Cases

```bash
# Generate synthetic test data with ground truth
python -m data.generate_targets
```

This creates three test cases in `data/test_cases/`:

- `test_case_a`: Ellipsoid (16 views)
- `test_case_b`: Bump (16 views)
- `test_case_c`: Cube (16 views)

### Run Optimization

```bash
# Quick test (50 iterations)
python main.py --config configs/config_quick_test.yaml

# Full optimization (500 iterations)
python main.py --config configs/config.yaml
```

### View Results

Results are saved to `results/[experiment_name]/`:

- `checkpoint_0050_100.obj` - Final optimized mesh
- `loss_curves_detailed.png` - Training visualization
- `evaluation_report.json` - Quantitative metrics
- `checkpoint_renders/` - Multi-view renderings

## Documentation

Detailed guides are available in the `docs/` directory:

- [Installation Guide](docs/INSTALLATION.md) - Step-by-step installation instructions
- [Usage Guide](docs/USAGE.md) - How to use the pipeline
- [Configuration Guide](docs/CONFIGURATION.md) - Parameter documentation
- [API Reference](docs/API.md) - Module and function documentation

## Project Structure

```
3d-model-reconstruction/
 src/                    # Core modules
    mesh.py            # Mesh operations and deformations
    camera.py          # Camera creation and utilities
    renderer_differentiable.py  # Soft rendering
    renderer_classic.py         # Hard rendering
    losses.py          # Loss functions
    optimizer.py       # Optimization loop
    utils.py           # Utilities and metrics
 configs/               # Configuration files
    config.yaml        # Main configuration
    config_quick_test.yaml  # Quick test config
 data/                  # Test case generation
    generate_targets.py
    test_cases/        # Generated test data
 tests/                 # Unit tests
 examples/              # Demo scripts
 docs/                  # Documentation
 main.py               # Main entry point
```

## How It Works

1. **Initialize**: Start with a simple sphere mesh
2. **Render**: Use differentiable rendering to create 2D images from 3D mesh
3. **Compare**: Calculate loss between rendered images and target images
4. **Optimize**: Update mesh vertices using gradient descent
5. **Repeat**: Iterate until mesh matches target

### Loss Functions

- **Silhouette Loss**: Measures difference between rendered and target silhouettes
- **Edge Loss**: Prevents irregular edge lengths
- **Laplacian Loss**: Enforces surface smoothness
- **Normal Consistency**: Maintains consistent surface normals

## Configuration

All experiments are controlled via YAML configuration files:

```yaml
# Example: configs/config.yaml
test_case: "test_case_c"

optimization:
  num_iterations: 500
  learning_rate: 0.01
  optimizer: "adam"

losses:
  silhouette_weight: 1.0
  edge_weight: 0.1
  laplacian_weight: 0.1
  adaptive_regularization: true

rendering:
  image_size: 256
  target_rendering_mode: "both"  # soft, hard, or both
```

See [CONFIGURATION.md](docs/CONFIGURATION.md) for detailed parameter descriptions.

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_renderer.py -v

# Run with coverage
pytest tests/ --cov=src
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch3D 0.7.5+
- CUDA-capable GPU (recommended, but CPU also supported)

See [requirements.txt](requirements.txt) for complete dependencies.

## Future Features

The following features are planned but not yet implemented:

- **Learning Rate Scheduling**: Decay learning rate during optimization
- **Early Stopping**: Automatically stop when convergence plateaus
- **Gradient Clipping**: Prevent exploding gradients
- **Initial Mesh Types**: Support ellipsoid and custom mesh initialization
- **Additional Camera Modes**: Random camera placement options

See configuration files for `[NOT IMPLEMENTED YET]` markers on specific parameters.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

- Built with [PyTorch3D](https://github.com/facebookresearch/pytorch3d)
- Inspired by research in differentiable rendering and inverse graphics
