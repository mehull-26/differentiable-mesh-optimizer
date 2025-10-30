# Configuration Reference

Complete reference for all configuration parameters in the 3D Model Reconstruction system.

## Configuration Files

- `configs/config.yaml` - Main configuration (500 iterations)
- `configs/config_quick_test.yaml` - Quick test (50 iterations)
- `data/generation_config.yaml` - Test case generation

## Main Configuration (`configs/config.yaml`)

### Test Case Selection

```yaml
test_case: "test_case_c"
```

**Options:**
- `test_case_a` - Ellipsoid (easy)
- `test_case_b` - Bump (medium)
- `test_case_c` - Cube (hard)

---

### Mesh Configuration

```yaml
mesh:
  subdivision_level: 2
  initial_type: "sphere"
```

**Parameters:**

- **subdivision_level** (int): Icosphere subdivision level
  - `0`: 12 vertices, 20 faces
  - `1`: 42 vertices, 80 faces
  - `2`: 162 vertices, 320 faces (recommended)
  - `3`: 642 vertices, 1280 faces
  - `4`: 2562 vertices, 5120 faces
  - Higher levels provide more detail but are slower and harder to optimize

- **initial_type** (str): Initial mesh shape
  - `sphere`: Regular sphere (default) ✅ **IMPLEMENTED**
  - `ellipsoid`: Pre-stretched sphere ⚠️ **NOT IMPLEMENTED YET**
  - `custom`: Load custom mesh (advanced) ⚠️ **NOT IMPLEMENTED YET**

---

### Optimization Settings

```yaml
optimization:
  num_iterations: 500
  learning_rate: 0.01
  optimizer: "adam"
  use_scheduler: false
  scheduler_type: "step"
  scheduler_gamma: 0.5
  scheduler_step_size: 200
```

**Parameters:**

- **num_iterations** (int): Total optimization steps
  - Quick test: 50
  - Standard: 500
  - High quality: 1000+

- **learning_rate** (float): Step size for gradient descent
  - Small (0.001-0.005): Stable, slow
  - Medium (0.01): Recommended
  - Large (0.05-0.1): Fast but unstable

- **optimizer** (str): Optimization algorithm
  - `adam`: Adaptive learning rate (recommended) ✅ **IMPLEMENTED**
  - `sgd`: Standard gradient descent ✅ **IMPLEMENTED**

- **use_scheduler** (bool): Enable learning rate decay ⚠️ **NOT IMPLEMENTED YET**
  
- **scheduler_type** (str): LR scheduler type ⚠️ **NOT IMPLEMENTED YET**
  - `step`: Multiply by gamma every N steps
  - `exponential`: Exponential decay
  - `cosine`: Cosine annealing

- **scheduler_gamma** (float): Decay factor (0.5 = half LR) ⚠️ **NOT IMPLEMENTED YET**

- **scheduler_step_size** (int): Steps between LR updates ⚠️ **NOT IMPLEMENTED YET**

---

### Loss Functions

```yaml
losses:
  silhouette_weight: 1.0
  edge_weight: 0.1
  laplacian_weight: 0.1
  normal_weight: 0.01
  adaptive_regularization: true
  silhouette_type: "mse"
```

**Parameters:**

- **silhouette_weight** (float): Data term weight
  - Measures how well rendered image matches target
  - Increase if reconstruction doesn't match targets
  
- **edge_weight** (float): Edge regularization weight
  - Penalizes irregular edge lengths
  - Prevents mesh distortion
  
- **laplacian_weight** (float): Smoothness weight
  - Enforces surface smoothness
  - Increase if mesh is too rough/spiky
  - Typical range: 0.1-0.5
  
- **normal_weight** (float): Normal consistency weight
  - Ensures smooth normal transitions
  - Usually small (0.01)
  
- **adaptive_regularization** (bool): Auto-scale regularization
  - When enabled, scales weights by sqrt(vertices/12)
  - Maintains smoothness at higher subdivision levels
  - **Recommended: true**
  
- **silhouette_type** (str): Silhouette loss function
  - `mse`: Mean squared error (L2) ✅ **IMPLEMENTED**
  - `bce`: Binary cross-entropy ✅ **IMPLEMENTED**

**Tuning tips:**
- Start with default weights
- If too rough: increase `laplacian_weight`
- If not matching: increase `silhouette_weight`
- If edges irregular: increase `edge_weight`

---

### Rendering Settings

```yaml
rendering:
  image_size: 256
  blur_radius: 0.01
  faces_per_pixel: 1
  target_rendering_mode: "both"
```

**Parameters:**

- **image_size** (int): Rendered image resolution
  - `64`: Very fast, low quality
  - `128`: Fast, good for testing
  - `256`: Recommended balance
  - `512`: High quality, slower, more memory

- **blur_radius** (float): Soft rendering blur amount
  - `0.001`: Sharp edges, weak gradients
  - `0.01`: Recommended balance
  - `0.1`: Very soft, strong gradients

- **faces_per_pixel** (int): Legacy parameter, not used ⚠️ **NOT USED**

- **target_rendering_mode** (str): Which targets to optimize against
  - `soft`: Soft-rendered targets only ✅ **IMPLEMENTED**
  - `hard`: Hard-rendered targets only ✅ **IMPLEMENTED**
  - `both`: Run both (2x time, compare results) ✅ **IMPLEMENTED**

---

### Visualization

```yaml
visualization:
  log_every: 50
  save_checkpoint_every: 100
  checkpoint_mode: "iteration"
  save_checkpoint_renders: true
  checkpoint_render_views: 4
  save_video: true
  video_fps: 10
  plot_loss_curves: true
  create_comparison: true
  verbose: true
```

**Parameters:**

- **log_every** (int): Print progress every N iterations

- **save_checkpoint_every** (int): Save mesh every N iterations (or %)

- **checkpoint_mode** (str): Checkpoint frequency type
  - `iteration`: Every N iterations
  - `percentage`: At 0%, 25%, 50%, 75%, 100%

- **save_checkpoint_renders** (bool): Save multi-view renders at checkpoints

- **checkpoint_render_views** (int): Number of viewing angles (4 or 8)

- **save_video** (bool): Create optimization animation (not implemented yet)

- **video_fps** (int): Video frame rate

- **plot_loss_curves** (bool): Generate loss plots

- **create_comparison** (bool): Create before/after comparison

- **verbose** (bool): Detailed console output

---

### Output Settings

```yaml
output:
  base_dir: "results"
  experiment_name: "default"
  save_intermediate: true
  save_history: true
```

**Parameters:**

- **base_dir** (str): Root directory for results

- **experiment_name** (str): Subfolder name
  - `null`: Auto-generate timestamp
  - `"my_exp"`: Use custom name

- **save_intermediate** (bool): Save checkpoint meshes

- **save_history** (bool): Save optimization_history.json

---

### Device Settings

```yaml
device:
  type: "cuda"
  cuda_device: 0
```

**Parameters:**

- **type** (str): Computation device
  - `auto`: Use GPU if available, else CPU
  - `cuda`: Use GPU (fail if unavailable)
  - `cpu`: Force CPU (slow)

- **cuda_device** (int): GPU device ID (0, 1, 2, ...)

---

### Advanced Options

```yaml
advanced:
  use_grad_clip: false
  grad_clip_value: 1.0
  use_early_stopping: false
  early_stop_patience: 50
  early_stop_threshold: 1e-6
  random_seed: 42
  use_vertex_bounds: true
  vertex_bound_min: -5.0
  vertex_bound_max: 5.0
```

**Parameters:**

- **use_grad_clip** (bool): Clip gradients to prevent explosion ⚠️ **NOT IMPLEMENTED YET**

- **grad_clip_value** (float): Maximum gradient norm ⚠️ **NOT IMPLEMENTED YET**

- **use_early_stopping** (bool): Stop if no improvement ⚠️ **NOT IMPLEMENTED YET**

- **early_stop_patience** (int): Wait N iterations before stopping ⚠️ **NOT IMPLEMENTED YET**

- **early_stop_threshold** (float): Minimum improvement threshold ⚠️ **NOT IMPLEMENTED YET**

- **random_seed** (int): Random seed for reproducibility ✅ **IMPLEMENTED**

- **use_vertex_bounds** (bool): Clamp vertex positions ✅ **IMPLEMENTED**

- **vertex_bound_min/max** (float): Vertex position bounds ✅ **IMPLEMENTED**

---

### Evaluation

```yaml
evaluation:
  compute_chamfer: true
  compute_normal_consistency: true
  compute_edge_length_variance: true
  save_evaluation_report: true
```

All boolean flags to enable/disable specific metrics.

---

## Generation Configuration (`data/generation_config.yaml`)

### Rendering Settings

```yaml
rendering:
  image_size: 256
  num_views: 16
  target_types:
    - hard
    - soft
  blur_radius_soft: 0.01
```

**Parameters:**

- **num_views** (int): Number of camera viewpoints (4, 8, 16, 32)

- **target_types** (list): Which target types to generate
  - `hard`: Binary silhouettes
  - `soft`: Blurred silhouettes

---

### Camera Configuration

```yaml
camera:
  fov: 60.0
  distance: 3.0
  azimuth_mode: uniform
  azimuth_range: [0, 360]
  elevation_mode: uniform
  elevation_range: [-75, 75]
```

**Parameters:**

- **fov** (float): Field of view in degrees (45-90 typical)

- **distance** (float): Camera distance from origin

- **azimuth_mode** (str): Horizontal angle distribution
  - `uniform`: Evenly spaced around object
  - `random`: Random angles

- **azimuth_range** (list): [min, max] in degrees

- **elevation_mode** (str): Vertical angle distribution
  - `uniform`: Evenly distributed from bottom to top
  - `fixed`: All cameras at same elevation
  - `mixed`: Varied pattern

- **elevation_range** (list): [min, max] in degrees
  - [-75, 75]: Complete coverage including bottom and top
  - [0, 60]: Only upper hemisphere
  - [-30, 30]: Equator region

---

### Test Case Definitions

```yaml
test_cases:
  test_case_a:
    description: "Sphere to Ellipsoid (Easy - uniform stretch)"
    difficulty: easy
    deformation:
      type: ellipsoid
      params:
        scale: [1.5, 1.0, 0.8]  # [x, y, z] stretch factors
```

**Deformation Types:**

1. **ellipsoid**
   ```yaml
   params:
     scale: [x, y, z]  # Axis scaling (1.0 = no change)
   ```

2. **bump**
   ```yaml
   params:
     center: [x, y, z]  # Bump location
     radius: 0.5        # Bump radius
     height: 0.3        # Bump amplitude
   ```

3. **twist**
   ```yaml
   params:
     angle: 45          # Twist angle in degrees
     axis: [0, 1, 0]    # Twist axis (usually Y)
   ```

4. **cube**
   ```yaml
   params:
     size: 1.0          # Cube edge length
   ```

---

## Example Configurations

### High Quality Reconstruction

```yaml
mesh:
  subdivision_level: 3  # More vertices

optimization:
  num_iterations: 1000
  learning_rate: 0.005  # Smaller steps

losses:
  silhouette_weight: 1.0
  laplacian_weight: 0.3  # More smoothness
  adaptive_regularization: true

rendering:
  image_size: 512  # Higher resolution
```

### Fast Testing

```yaml
mesh:
  subdivision_level: 1  # Fewer vertices

optimization:
  num_iterations: 50
  learning_rate: 0.02  # Bigger steps

rendering:
  image_size: 128
  target_rendering_mode: "soft"  # Only one mode
```

### CPU-Friendly

```yaml
rendering:
  image_size: 128  # Smaller images

mesh:
  subdivision_level: 1  # Fewer vertices

device:
  type: "cpu"

visualization:
  save_checkpoint_renders: false  # Skip extra rendering
```

---

## Tips for Tuning

1. **Start with defaults** - They work well for most cases

2. **Change one thing at a time** - Easier to see effects

3. **Quick iterations** - Test with `config_quick_test.yaml` first

4. **Monitor loss curves** - Check `loss_curves_detailed.png`

5. **Compare experiments** - Run with different configs, compare results

---

## See Also

- [USAGE.md](USAGE.md) - How to run experiments
- [INSTALLATION.md](INSTALLATION.md) - Setup instructions
- [API.md](API.md) - Programmatic usage
