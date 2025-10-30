# Usage Guide

This guide explains how to use the 3D Model Reconstruction system for various tasks.

## Table of Contents

- [Basic Workflow](#basic-workflow)
- [Generating Test Cases](#generating-test-cases)
- [Running Optimization](#running-optimization)
- [Understanding Results](#understanding-results)
- [Customizing Experiments](#customizing-experiments)
- [Advanced Usage](#advanced-usage)

## Basic Workflow

The typical workflow consists of three steps:

1. **Generate test cases** - Create synthetic data with ground truth
2. **Run optimization** - Reconstruct 3D shape from 2D silhouettes
3. **Analyze results** - Evaluate reconstruction quality

## Generating Test Cases

### Quick Generation

Generate all three test cases with default settings:

```bash
python -m data.generate_targets
```

This creates:
- `data/test_cases/test_case_a/` - Ellipsoid (Easy)
- `data/test_cases/test_case_b/` - Bump (Medium)
- `data/test_cases/test_case_c/` - Cube (Hard)

### Custom Generation

Edit `data/generation_config.yaml` to customize:

```yaml
rendering:
  num_views: 16        # Number of camera views
  image_size: 256      # Image resolution

camera:
  elevation_mode: uniform    # Camera distribution
  elevation_range: [-75, 75] # Viewing angle range

mesh:
  subdivision_level: 2  # Mesh complexity (0-5)
```

Then regenerate:

```bash
python -m data.generate_targets
```

### Test Case Details

Each test case includes:
- `ground_truth.obj` - Target mesh to reconstruct
- `initial.obj` - Starting sphere mesh
- `camera_config.json` - Camera parameters
- `metadata.json` - Test case information
- `targets_hard/` - Binary silhouettes (16 images)
- `targets_soft/` - Soft silhouettes (16 images)
- `visualization.png` - All views in grid

## Running Optimization

### Quick Test (50 iterations, ~30 seconds)

```bash
python main.py --config configs/config_quick_test.yaml
```

Output location: `results/quick_test/`

### Full Optimization (500 iterations, ~5 minutes)

```bash
python main.py --config configs/config.yaml
```

Output location: `results/default/`

### Custom Experiment

```bash
python main.py --config configs/config.yaml --experiment my_experiment
```

Output location: `results/my_experiment/`

### Monitoring Progress

The optimization displays real-time progress:

```
Iteration 100/500
  Total loss:      0.042
  Silhouette loss: 0.004
  Edge loss:       0.002
  Laplacian loss:  0.007
  Avg displacement: 0.29
```

- **Total loss**: Combined objective (lower is better)
- **Silhouette loss**: How well rendered matches target
- **Edge/Laplacian loss**: Regularization terms (smoothness)
- **Displacement**: How much vertices moved

## Understanding Results

### Output Files

After optimization completes, find results in `results/[experiment_name]/`:

#### Meshes
- `checkpoint_0000_00.obj` - Initial sphere
- `checkpoint_NNNN_PP.obj` - Intermediate checkpoints
- `checkpoint_0500_100.obj` - Final optimized mesh
- `reference_ground_truth.obj` - Target mesh (for comparison)

#### Visualizations
- `checkpoint_renders/checkpoint_0000_00.png` - Initial renders (8 views)
- `checkpoint_renders/checkpoint_0500_100.png` - Final renders (8 views)
- `checkpoint_renders/reference_ground_truth.png` - Target renders
- `loss_curves_detailed.png` - Training curves (loss over time)

#### Metrics
- `optimization_history.json` - Complete training log
- `evaluation_report.json` - Final evaluation metrics
- `config.yaml` - Experiment configuration

### Evaluation Metrics

Open `evaluation_report.json` to see:

```json
{
  "chamfer_distance": 0.102,
  "normal_consistency": 0.997,
  "3d_vertex_matching": {
    "aligned_rmse": 0.112,
    "aligned_mae": 0.101,
    "vertices_within_100mm": 41.4
  }
}
```

**Metrics explained:**
- **Chamfer Distance**: Average closest-point distance (lower is better)
- **Normal Consistency**: Surface orientation similarity (higher is better, max 1.0)
- **RMSE/MAE**: Root mean square / mean absolute error of vertex positions
- **Vertices within X mm**: Percentage of vertices close to ground truth

### Viewing Meshes

Open `.obj` files in 3D software:
- **MeshLab** (free, cross-platform)
- **Blender** (free, powerful)
- **Online**: [https://3dviewer.net/](https://3dviewer.net/)

## Customizing Experiments

### Change Test Case

Edit `configs/config.yaml`:

```yaml
test_case: "test_case_a"  # Options: test_case_a, test_case_b, test_case_c
```

### Adjust Optimization Settings

```yaml
optimization:
  num_iterations: 500      # More = better (but slower)
  learning_rate: 0.01      # Lower = more stable, higher = faster
  optimizer: "adam"        # Options: adam, sgd
```

### Tune Loss Weights

```yaml
losses:
  silhouette_weight: 1.0   # Data term (how well it matches)
  edge_weight: 0.1         # Edge regularization
  laplacian_weight: 0.1    # Smoothness
  normal_weight: 0.01      # Normal consistency
  adaptive_regularization: true  # Auto-scale for mesh size
```

**Tip**: If mesh is too rough, increase `laplacian_weight`. If not matching targets well, increase `silhouette_weight`.

### Change Rendering Mode

```yaml
rendering:
  target_rendering_mode: "both"  # Options: soft, hard, both
```

- **soft**: Use soft-rendered targets (blurred edges)
- **hard**: Use hard-rendered targets (binary edges)
- **both**: Run separate optimizations for each

### Adjust Mesh Resolution

```yaml
mesh:
  subdivision_level: 2  # Options: 0 (12v), 1 (42v), 2 (162v), 3 (642v), 4 (2562v)
```

Higher = more detail but slower and harder to optimize.

## Advanced Usage

### Running Specific Test Case

```bash
python main.py --config configs/config.yaml --test-case test_case_b
```

### Batch Processing

```bash
# Run all test cases
for case in test_case_a test_case_b test_case_c; do
    python main.py --config configs/config.yaml --experiment $case
done
```

### Resume from Checkpoint

Currently not supported. To continue optimization:
1. Increase `num_iterations` in config
2. Rerun from scratch (fast with GPU)

### Export for Visualization

Convert OBJ to other formats:

```python
# Using trimesh library
import trimesh
mesh = trimesh.load('results/my_exp/checkpoint_0500_100.obj')
mesh.export('output.ply')  # PLY format
mesh.export('output.stl')  # STL format
```

### Custom Loss Functions

Edit `src/losses.py` to add new losses:

```python
def my_custom_loss(vertices, faces, ...):
    # Your loss computation
    return loss_value
```

Then add to `src/optimizer.py`:

```python
custom_loss = my_custom_loss(vertices, faces, ...)
total_loss += custom_loss * self.weights.get('custom', 0.1)
```

### Using Your Own Data

To use real images instead of synthetic:

1. Create directory structure:
   ```
   data/test_cases/my_case/
   ├── camera_config.json
   ├── targets_hard/
   │   ├── view_00.png
   │   └── ...
   └── ground_truth.obj (optional, for evaluation)
   ```

2. Update `camera_config.json` with your camera parameters

3. Run optimization:
   ```bash
   python main.py --config configs/config.yaml --test-case my_case
   ```

## Common Use Cases

### Use Case 1: Quick Shape Test

Want to test if a shape is reconstructible?

```bash
# 1. Edit test case in generation_config.yaml
# 2. Generate
python -m data.generate_targets

# 3. Quick test
python main.py --config configs/config_quick_test.yaml
```

### Use Case 2: High-Quality Reconstruction

Need best possible quality?

```yaml
# configs/config.yaml
optimization:
  num_iterations: 1000  # Increase iterations
  
losses:
  silhouette_weight: 1.0
  laplacian_weight: 0.2  # Increase smoothness
  
mesh:
  subdivision_level: 3  # More vertices (642)
```

### Use Case 3: Compare Soft vs Hard Rendering

Which works better for your shape?

```yaml
rendering:
  target_rendering_mode: "both"
```

Then compare results in `results/*/soft_target/` vs `results/*/hard_target/`.

## Performance Tips

### GPU Optimization
- Increase `image_size` to 512 for better quality (if memory allows)
- Use `target_rendering_mode: "soft"` for faster single run
- Close other GPU applications

### CPU Optimization
- Reduce `image_size` to 128
- Reduce `num_views` in generation
- Use `subdivision_level: 1` for faster mesh

### Memory Management
If out of memory:
```yaml
rendering:
  image_size: 128       # Reduce from 256
mesh:
  subdivision_level: 1  # Reduce from 2
```

## Troubleshooting

### Optimization not converging
- Increase `num_iterations`
- Reduce `learning_rate` (try 0.001)
- Increase `silhouette_weight`

### Mesh too rough/spiky
- Increase `laplacian_weight` (try 0.2 or 0.5)
- Enable `adaptive_regularization: true`
- Use lower `subdivision_level`

### Mesh exploding/NaN loss
- Reduce `learning_rate`
- Enable `use_vertex_bounds: true`
- Check camera configuration

## Next Steps

- Explore [CONFIGURATION.md](CONFIGURATION.md) for all parameters
- Check [API.md](API.md) for programmatic usage
- See [examples/](../examples/) for demo scripts

## Getting Help

For issues or questions:
1. Check this guide and [INSTALLATION.md](INSTALLATION.md)
2. Review [example configs](../configs/)
3. Open an issue on GitHub with:
   - Config file used
   - Command run
   - Error message or unexpected behavior
