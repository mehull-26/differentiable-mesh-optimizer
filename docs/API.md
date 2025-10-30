# API Reference

Documentation for the core modules and their APIs.

## Module Overview

```
src/
├── mesh.py              # Mesh creation and manipulation
├── camera.py            # Camera setup and configuration
├── renderer_differentiable.py  # Soft rendering
├── renderer_classic.py  # Hard rendering  
├── renderer.py          # Unified renderer interface
├── losses.py            # Loss functions
├── optimizer.py         # Optimization loop
└── utils.py             # Utilities and evaluation
```

## Core Modules

### mesh.py

#### `create_icosphere(subdivisions=2)`

Create an icosphere mesh.

**Parameters:**
- `subdivisions` (int): Subdivision level (0-5)

**Returns:**
- `vertices` (torch.Tensor): Shape (V, 3)
- `faces` (torch.Tensor): Shape (F, 3)

**Example:**
```python
from src.mesh import create_icosphere

vertices, faces = create_icosphere(subdivisions=2)
# vertices: (162, 3), faces: (320, 3)
```

#### `create_deformed_sphere(deformation_type, params, subdivisions=2)`

Create a deformed sphere.

**Parameters:**
- `deformation_type` (str): 'ellipsoid', 'bump', 'twist', or 'cube'
- `params` (dict): Deformation parameters
- `subdivisions` (int): Subdivision level

**Returns:**
- `vertices, faces` (torch.Tensor)

**Example:**
```python
from src.mesh import create_deformed_sphere

# Ellipsoid
vertices, faces = create_deformed_sphere(
    deformation_type='ellipsoid',
    params={'scale': [1.5, 1.0, 0.8]}
)

# Cube
vertices, faces = create_deformed_sphere(
    deformation_type='cube',
    params={'size': 1.0}
)
```

#### `save_mesh(vertices, faces, path)`

Save mesh to OBJ file.

#### `load_mesh(path)`

Load mesh from OBJ file.

---

### camera.py

#### `create_camera(azimuth, elevation, distance, image_size=256, device='cuda')`

Create a single camera.

**Parameters:**
- `azimuth` (float): Horizontal angle in degrees
- `elevation` (float): Vertical angle in degrees
- `distance` (float): Camera distance from origin
- `image_size` (int): Image resolution
- `device` (str): 'cuda' or 'cpu'

**Returns:**
- PyTorch3D FoVPerspectiveCameras object

**Example:**
```python
from src.camera import create_camera

camera = create_camera(
    azimuth=45,
    elevation=30,
    distance=3.0,
    image_size=256,
    device='cuda'
)
```

#### `create_camera_circle(num_views=8, elevation=30, distance=3.0, ...)`

Create multiple cameras in a circle.

**Returns:**
- List of camera objects

---

### renderer_differentiable.py

#### `DifferentiableRenderer(image_size=256, blur_radius=0.01, device='cuda')`

Soft renderer with gradient flow.

**Methods:**

##### `render_silhouette(vertices, faces, cameras)`

Render soft silhouette.

**Returns:**
- `silhouette` (torch.Tensor): Shape (1, H, W, 1), values in [0, 1]

**Example:**
```python
from src.renderer_differentiable import DifferentiableRenderer
from src.mesh import create_icosphere
from src.camera import create_camera

renderer = DifferentiableRenderer(
    image_size=256,
    blur_radius=0.01,
    device='cuda'
)

vertices, faces = create_icosphere(subdivisions=1)
camera = create_camera(45, 30, 3.0, device='cuda')

silhouette = renderer.render_silhouette(vertices, faces, camera)
# silhouette: (1, 256, 256, 1)
```

##### `render_depth(vertices, faces, cameras)`

Render depth map.

##### `render_rgb(vertices, faces, cameras, lights=None, vertex_colors=None)`

Render RGB image with lighting.

---

### renderer_classic.py

#### `ClassicRenderer(image_size=256, device='cuda')`

Binary renderer for hard silhouettes.

**Methods:**

##### `render_silhouette(vertices, faces, cameras)`

Render binary silhouette (sharp edges).

**Example:**
```python
from src.renderer_classic import ClassicRenderer

renderer = ClassicRenderer(image_size=256, device='cuda')
silhouette = renderer.render_silhouette(vertices, faces, camera)
# Binary values: 0.0 or 1.0
```

---

### losses.py

#### `silhouette_loss(rendered, target, loss_type='mse')`

Compute silhouette matching loss.

**Parameters:**
- `rendered` (torch.Tensor): Rendered silhouette
- `target` (torch.Tensor): Target silhouette
- `loss_type` (str): 'mse' or 'bce'

**Returns:**
- Loss value (torch.Tensor)

#### `edge_loss(vertices, faces)`

Edge length regularization.

#### `laplacian_loss(vertices, faces)`

Laplacian smoothness loss.

#### `normal_consistency_loss(vertices, faces)`

Normal consistency regularization.

**Example:**
```python
from src.losses import silhouette_loss, edge_loss

sil_loss = silhouette_loss(rendered, target, loss_type='mse')
edge_reg = edge_loss(vertices, faces)
total_loss = sil_loss + 0.1 * edge_reg
```

---

### optimizer.py

#### `MeshOptimizer(renderer, device='cuda')`

Main optimization class.

**Methods:**

##### `optimize_mesh(initial_vertices, faces, target_images, cameras, num_iterations=500, learning_rate=0.01, ...)`

Run optimization loop.

**Parameters:**
- `initial_vertices` (torch.Tensor): Starting mesh vertices
- `faces` (torch.Tensor): Triangle indices
- `target_images` (list): List of target silhouettes
- `cameras` (list): List of camera objects
- `num_iterations` (int): Optimization steps
- `learning_rate` (float): Step size
- `weights` (dict): Loss weights

**Returns:**
- `optimized_vertices` (torch.Tensor): Final mesh vertices
- `history` (dict): Optimization metrics

**Example:**
```python
from src.optimizer import MeshOptimizer
from src.renderer_differentiable import DifferentiableRenderer

renderer = DifferentiableRenderer(device='cuda')
optimizer = MeshOptimizer(renderer=renderer, device='cuda')

optimized_vertices, history = optimizer.optimize_mesh(
    initial_vertices=vertices,
    faces=faces,
    target_images=targets,
    cameras=cameras,
    num_iterations=500,
    learning_rate=0.01,
    weights={
        'silhouette': 1.0,
        'edge': 0.1,
        'laplacian': 0.1,
        'normal': 0.01
    }
)
```

---

### utils.py

#### `compute_chamfer_distance(vertices1, faces1, vertices2, faces2, num_samples=1000)`

Compute Chamfer distance between two meshes.

**Returns:**
- Dictionary with distance metrics

#### `compute_normal_consistency(vertices1, faces1, vertices2, faces2)`

Compute normal consistency metric.

**Returns:**
- Float value (0-1, higher is better)

#### `compute_3d_vertex_matching(optimized_vertices, ground_truth_vertices, alignment='procrustes')`

Compute 3D vertex matching with alignment.

**Parameters:**
- `alignment` (str): 'center', 'procrustes', or 'none'

**Returns:**
- Dictionary with RMSE, MAE, and other metrics

---

## Programmatic Usage Examples

### Example 1: Basic Reconstruction

```python
import torch
from src.mesh import create_icosphere, create_deformed_sphere
from src.camera import create_camera_circle
from src.renderer_differentiable import DifferentiableRenderer
from src.optimizer import MeshOptimizer

# Create ground truth
gt_vertices, gt_faces = create_deformed_sphere(
    deformation_type='ellipsoid',
    params={'scale': [1.5, 1.0, 0.8]}
)

# Setup cameras
cameras = create_camera_circle(
    num_views=8,
    elevation=30,
    distance=3.0,
    device='cuda'
)

# Render targets
renderer = DifferentiableRenderer(device='cuda')
targets = []
for cam in cameras:
    sil = renderer.render_silhouette(gt_vertices, gt_faces, cam)
    targets.append(sil)

# Initialize mesh
init_vertices, init_faces = create_icosphere(subdivisions=2)

# Optimize
optimizer = MeshOptimizer(renderer=renderer, device='cuda')
optimized_vertices, history = optimizer.optimize_mesh(
    initial_vertices=init_vertices,
    faces=init_faces,
    target_images=targets,
    cameras=cameras,
    num_iterations=500,
    learning_rate=0.01,
    weights={'silhouette': 1.0, 'edge': 0.1, 'laplacian': 0.1}
)

# Save result
from src.mesh import save_mesh
save_mesh(optimized_vertices, init_faces, 'result.obj')
```

### Example 2: Custom Loss Weights

```python
# Different weight configurations
weights_smooth = {
    'silhouette': 1.0,
    'edge': 0.1,
    'laplacian': 0.5,  # More smoothness
    'normal': 0.02
}

weights_accurate = {
    'silhouette': 2.0,  # Prioritize matching
    'edge': 0.05,
    'laplacian': 0.05,
    'normal': 0.01
}

# Run with custom weights
optimized_vertices, _ = optimizer.optimize_mesh(
    ...,
    weights=weights_smooth
)
```

### Example 3: Evaluation

```python
from src.utils import (
    compute_chamfer_distance,
    compute_normal_consistency,
    compute_3d_vertex_matching
)

# Chamfer distance
chamfer = compute_chamfer_distance(
    optimized_vertices, init_faces,
    gt_vertices, gt_faces
)
print(f"Chamfer: {chamfer['chamfer_distance']:.4f}")

# Normal consistency
normal_cons = compute_normal_consistency(
    optimized_vertices, init_faces,
    gt_vertices, gt_faces
)
print(f"Normal consistency: {normal_cons:.4f}")

# 3D matching
matching = compute_3d_vertex_matching(
    optimized_vertices,
    gt_vertices,
    alignment='procrustes'
)
print(f"RMSE: {matching['aligned_rmse']:.4f}")
```

---

## Type Hints

```python
# Common types
Vertices = torch.Tensor  # (V, 3) float32
Faces = torch.Tensor     # (F, 3) int64
Image = torch.Tensor     # (1, H, W, C) float32
Camera = pytorch3d.renderer.cameras.FoVPerspectiveCameras
```

---

## Error Handling

All functions raise standard Python exceptions:
- `ValueError`: Invalid parameters
- `FileNotFoundError`: Missing files
- `RuntimeError`: CUDA/computation errors

**Example:**
```python
try:
    vertices, faces = create_icosphere(subdivisions=10)
except ValueError as e:
    print(f"Invalid subdivision level: {e}")
```

---

## Performance Notes

- **GPU highly recommended**: 160-240x faster than CPU
- **Batch rendering**: Process multiple views in parallel
- **Memory scaling**: image_size and subdivision_level are main factors

---

## See Also

- [USAGE.md](USAGE.md) - High-level usage guide
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration parameters
- [examples/](../examples/) - Demo scripts
