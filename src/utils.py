"""
Visualization and Analysis Utilities

This module provides comprehensive tools for:
1. Loading and processing data
2. Visualizing optimization progress
3. Comparing meshes and computing metrics
4. Creating videos and animations
5. Generating reports

These tools help understand what's happening during optimization
and evaluate the quality of results.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import json
from pathlib import Path
from PIL import Image
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings


# ==================== DATA LOADING ====================

def load_target_images(test_case_dir: Path,
                       image_size: int = 256,
                       target_type: str = 'soft') -> List[torch.Tensor]:
    """
    Load all target images from a test case.

    Args:
        test_case_dir: Path to test case directory
        image_size: Resize images to this size
        target_type: Type of targets to load ('soft' or 'hard')

    Returns:
        List of target image tensors [1, H, W, 1]
    """
    # Load from targets_soft/ or targets_hard/ directory
    targets_dir = test_case_dir / f"targets_{target_type}"

    # Fallback to legacy 'targets/' directory if specific type not found
    if not targets_dir.exists():
        targets_dir = test_case_dir / "targets"
        if not targets_dir.exists():
            raise ValueError(
                f"Targets directory not found: {targets_dir} or {test_case_dir / f'targets_{target_type}'}")

    target_files = sorted(targets_dir.glob("view_*.png"))

    if len(target_files) == 0:
        raise ValueError(f"No target images found in {targets_dir}")

    images = []
    for img_file in target_files:
        # Load as grayscale
        img = Image.open(img_file).convert('L')
        img = img.resize((image_size, image_size))

        # Convert to tensor [1, H, W, 1]
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(-1)
        images.append(img_tensor)

    return images


def save_experiment_config(config: dict, output_dir: Path):
    """
    Save experiment configuration to output directory.

    Args:
        config: Configuration dictionary
        output_dir: Output directory
    """
    import yaml

    config_path = output_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"✓ Saved configuration: {config_path}")


def setup_experiment_directory(config: dict) -> Path:
    """
    Setup output directory for experiment.

    Args:
        config: Configuration dictionary

    Returns:
        Path to experiment directory
    """
    base_dir = Path(config['output']['base_dir'])
    experiment_name = config['output'].get('experiment_name')

    if experiment_name is None or experiment_name == 'default':
        # Auto-generate experiment name with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        test_case = config['test_case'].split('_')[-1]  # Extract 'a', 'b', 'c'
        experiment_name = f"exp_{test_case}_{timestamp}"

    output_dir = base_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


# ==================== VISUALIZATION ====================

def visualize_optimization(history: Dict, output_dir: Path):
    """
    Create comprehensive visualization of optimization progress.

    This creates multiple plots:
    1. Total loss over time
    2. Individual loss components
    3. Loss reduction percentage

    Args:
        history: Optimization history dict with loss arrays
        output_dir: Where to save plots
    """
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    iterations = range(len(history['total_loss']))

    # 1. Total loss
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(iterations, history['total_loss'], linewidth=2, color='#2E86AB')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Total Loss', fontsize=12)
    ax1.set_title('Total Loss Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add initial and final loss annotations
    initial_loss = history['total_loss'][0]
    final_loss = history['total_loss'][-1]
    reduction = (1 - final_loss/initial_loss) * 100

    ax1.annotate(f'Initial: {initial_loss:.4f}',
                 xy=(0, initial_loss),
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 fontsize=10)

    ax1.annotate(f'Final: {final_loss:.4f}\nReduction: {reduction:.1f}%',
                 xy=(len(iterations)-1, final_loss),
                 xytext=(-80, -20), textcoords='offset points',
                 bbox=dict(boxstyle='round',
                           facecolor='lightgreen', alpha=0.5),
                 fontsize=10)

    # 2. Individual loss components
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(iterations, history['silhouette_loss'],
             label='Silhouette', linewidth=2)
    ax2.plot(iterations, history['edge_loss'], label='Edge', linewidth=2)
    ax2.plot(iterations, history['laplacian_loss'],
             label='Laplacian', linewidth=2)
    if 'normal_loss' in history and any(v > 0 for v in history['normal_loss']):
        ax2.plot(iterations, history['normal_loss'],
                 label='Normal', linewidth=2)

    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Individual Loss Components', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. Silhouette loss (zoomed)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(iterations, history['silhouette_loss'],
             linewidth=2, color='#A23B72')
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Silhouette Loss', fontsize=12)
    ax3.set_title('Silhouette Loss (Data Term)',
                  fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Regularization losses (zoomed)
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(iterations, history['edge_loss'], label='Edge', linewidth=2)
    ax4.plot(iterations, history['laplacian_loss'],
             label='Laplacian', linewidth=2)
    ax4.set_xlabel('Iteration', fontsize=12)
    ax4.set_ylabel('Regularization Loss', fontsize=12)
    ax4.set_title('Regularization Terms', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Save figure
    output_path = output_dir / "loss_curves_detailed.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved detailed loss curves: {output_path}")


def compare_meshes(optimized_vertices: torch.Tensor,
                   ground_truth_vertices: torch.Tensor,
                   faces: torch.Tensor,
                   gt_faces: torch.Tensor = None,
                   compute_chamfer: bool = True,
                   compute_normal_consistency: bool = True,
                   compute_edge_variance: bool = True) -> Dict:
    """
    Compare optimized mesh with ground truth.

    Computes various metrics to quantify reconstruction quality:
    1. Chamfer distance - point-to-point distance
    2. Normal consistency - alignment of normals
    3. Edge length variance - mesh regularity

    Args:
        optimized_vertices: Optimized mesh vertices (V1, 3)
        ground_truth_vertices: Ground truth vertices (V2, 3)
        faces: Triangle faces for optimized mesh (F1, 3)
        gt_faces: Triangle faces for ground truth mesh (F2, 3), if None uses faces
        compute_chamfer: Compute Chamfer distance
        compute_normal_consistency: Compute normal consistency
        compute_edge_variance: Compute edge length variance

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Use same faces for GT if not provided
    if gt_faces is None:
        gt_faces = faces

    # 1. Chamfer Distance
    if compute_chamfer:
        # Chamfer distance: average nearest neighbor distance both ways
        # For each optimized vertex, find nearest GT vertex
        dists_opt_to_gt = torch.cdist(
            optimized_vertices, ground_truth_vertices)
        min_dists_opt_to_gt = dists_opt_to_gt.min(dim=1)[0]

        # For each GT vertex, find nearest optimized vertex
        min_dists_gt_to_opt = dists_opt_to_gt.min(dim=0)[0]

        # Chamfer distance is the sum of both directions
        chamfer_dist = (min_dists_opt_to_gt.mean() +
                        min_dists_gt_to_opt.mean()).item()

        metrics['chamfer_distance'] = chamfer_dist
        metrics['chamfer_opt_to_gt'] = min_dists_opt_to_gt.mean().item()
        metrics['chamfer_gt_to_opt'] = min_dists_gt_to_opt.mean().item()

        print(f"\nChamfer Distance: {chamfer_dist:.6f}")
        print(f"  - Optimized → GT: {metrics['chamfer_opt_to_gt']:.6f}")
        print(f"  - GT → Optimized: {metrics['chamfer_gt_to_opt']:.6f}")

    # 2. Normal Consistency
    if compute_normal_consistency:
        # Compute normals for both meshes with their respective faces
        opt_normals = compute_vertex_normals(optimized_vertices, faces)
        gt_normals = compute_vertex_normals(ground_truth_vertices, gt_faces)

        # For different sized meshes, compare using nearest neighbors
        if optimized_vertices.shape[0] != ground_truth_vertices.shape[0]:
            # Find nearest vertex in GT for each optimized vertex
            dists = torch.cdist(optimized_vertices, ground_truth_vertices)
            nearest_gt_idx = dists.argmin(dim=1)
            gt_normals_matched = gt_normals[nearest_gt_idx]

            # Cosine similarity (dot product of normalized normals)
            normal_similarity = (
                opt_normals * gt_normals_matched).sum(dim=1).mean().item()
        else:
            # Same size - direct comparison
            normal_similarity = (
                opt_normals * gt_normals).sum(dim=1).mean().item()

        metrics['normal_consistency'] = normal_similarity

        print(f"Normal Consistency: {normal_similarity:.6f} (1.0 = perfect)")

    # 3. Edge Length Variance
    if compute_edge_variance:
        opt_variance = compute_edge_length_variance(optimized_vertices, faces)
        gt_variance = compute_edge_length_variance(
            ground_truth_vertices, gt_faces)

        metrics['edge_variance_optimized'] = opt_variance
        metrics['edge_variance_ground_truth'] = gt_variance

        print(f"Edge Length Variance:")
        print(f"  - Optimized: {opt_variance:.6f}")
        print(f"  - Ground Truth: {gt_variance:.6f}")

    # 4. Vertex Displacement Statistics
    if optimized_vertices.shape == ground_truth_vertices.shape:
        displacement = (optimized_vertices - ground_truth_vertices).norm(dim=1)
        metrics['mean_vertex_displacement'] = displacement.mean().item()
        metrics['max_vertex_displacement'] = displacement.max().item()
        metrics['std_vertex_displacement'] = displacement.std().item()

        print(f"Vertex Displacement:")
        print(f"  - Mean: {metrics['mean_vertex_displacement']:.6f}")
        print(f"  - Max: {metrics['max_vertex_displacement']:.6f}")
        print(f"  - Std: {metrics['std_vertex_displacement']:.6f}")

    return metrics


def compute_3d_vertex_matching(optimized_vertices: torch.Tensor,
                               ground_truth_vertices: torch.Tensor,
                               alignment: str = 'procrustes') -> Dict:
    """
    Compute 3D vertex matching metrics after optimal alignment.

    This performs alignment (translation, rotation, optional scale) to best match
    the ground truth, then computes vertex-wise distances in 3D space.

    Args:
        optimized_vertices: Optimized mesh vertices (V, 3)
        ground_truth_vertices: Ground truth vertices (V, 3) - must be same size
        alignment: Type of alignment ('procrustes', 'center', 'none')

    Returns:
        Dictionary with:
        - aligned_rmse: Root mean squared error after alignment
        - aligned_mae: Mean absolute error after alignment
        - aligned_max_error: Maximum vertex error after alignment
        - vertex_match_percentage_XXmm: % of vertices within XX mm threshold
        - per_vertex_errors: Individual vertex errors (V,)
    """
    metrics = {}

    # Check if vertices have same count
    if optimized_vertices.shape[0] != ground_truth_vertices.shape[0]:
        print(f"\n⚠ WARNING: Vertex count mismatch!")
        print(
            f"  Optimized: {optimized_vertices.shape[0]}, GT: {ground_truth_vertices.shape[0]}")
        print(f"  Using nearest neighbor matching instead of direct alignment.")

        # Use nearest neighbor matching
        dists = torch.cdist(optimized_vertices, ground_truth_vertices)
        min_dists = dists.min(dim=1)[0]

        metrics['aligned_rmse'] = torch.sqrt((min_dists ** 2).mean()).item()
        metrics['aligned_mae'] = min_dists.mean().item()
        metrics['aligned_max_error'] = min_dists.max().item()
        # Don't include per_vertex_errors in JSON output (too large)
        # metrics['per_vertex_errors'] = min_dists.cpu().numpy()
    else:
        # Same vertex count - perform alignment
        opt_verts = optimized_vertices.clone()
        gt_verts = ground_truth_vertices.clone()

        if alignment == 'center' or alignment == 'procrustes':
            # Center both meshes
            opt_center = opt_verts.mean(dim=0, keepdim=True)
            gt_center = gt_verts.mean(dim=0, keepdim=True)
            opt_verts = opt_verts - opt_center
            gt_verts = gt_verts - gt_center

        if alignment == 'procrustes':
            # Procrustes alignment: find optimal rotation
            # Using SVD: R = V @ U^T where U, S, V = SVD(H)
            # H = gt_verts^T @ opt_verts
            H = gt_verts.T @ opt_verts
            U, S, Vh = torch.linalg.svd(H)
            R = U @ Vh

            # Ensure proper rotation (det(R) = 1)
            if torch.det(R) < 0:
                Vh[-1, :] *= -1
                R = U @ Vh

            # Apply rotation to optimized vertices
            opt_verts = opt_verts @ R.T

        # Compute vertex-wise errors
        per_vertex_errors = (opt_verts - gt_verts).norm(dim=1)

        # Metrics
        metrics['aligned_rmse'] = torch.sqrt(
            (per_vertex_errors ** 2).mean()).item()
        metrics['aligned_mae'] = per_vertex_errors.mean().item()
        metrics['aligned_max_error'] = per_vertex_errors.max().item()
        metrics['aligned_median_error'] = per_vertex_errors.median().item()
        # Don't include per_vertex_errors in JSON output (too large, 162+ values)
        # Store summary statistics instead
        metrics['aligned_std_error'] = per_vertex_errors.std().item()
        metrics['aligned_min_error'] = per_vertex_errors.min().item()
        # metrics['per_vertex_errors'] = per_vertex_errors.cpu().numpy()

        # Percentage of vertices within distance thresholds
        for threshold in [0.01, 0.05, 0.1, 0.2]:
            within = (per_vertex_errors <
                      threshold).float().mean().item() * 100
            metrics[f'vertices_within_{int(threshold*1000)}mm'] = within

    # Print results
    print(f"\n3D Vertex Matching (after {alignment} alignment):")
    print(f"  RMSE: {metrics['aligned_rmse']:.6f}")
    print(f"  MAE:  {metrics['aligned_mae']:.6f}")
    print(f"  Max error: {metrics['aligned_max_error']:.6f}")
    if 'aligned_median_error' in metrics:
        print(f"  Median error: {metrics['aligned_median_error']:.6f}")

    print(f"\nVertex accuracy (% within threshold):")
    for key in sorted(metrics.keys()):
        if 'vertices_within' in key:
            threshold_mm = int(key.split('_')[-1].replace('mm', ''))
            print(
                f"  Within {threshold_mm/1000:.3f} units: {metrics[key]:.1f}%")

    return metrics


def compute_vertex_normals(vertices: torch.Tensor,
                           faces: torch.Tensor) -> torch.Tensor:
    """
    Compute vertex normals by averaging face normals.

    Args:
        vertices: Vertex positions (V, 3)
        faces: Triangle faces (F, 3)

    Returns:
        Vertex normals (V, 3)
    """
    # Get triangle vertices
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Compute face normals via cross product
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)

    # Normalize
    face_normals = face_normals / \
        (face_normals.norm(dim=1, keepdim=True) + 1e-8)

    # Accumulate face normals to vertices
    vertex_normals = torch.zeros_like(vertices)
    for i in range(3):
        vertex_normals.index_add_(0, faces[:, i], face_normals)

    # Normalize
    vertex_normals = vertex_normals / \
        (vertex_normals.norm(dim=1, keepdim=True) + 1e-8)

    return vertex_normals


def compute_edge_length_variance(vertices: torch.Tensor,
                                 faces: torch.Tensor) -> float:
    """
    Compute variance of edge lengths (measures mesh regularity).

    Args:
        vertices: Vertex positions (V, 3)
        faces: Triangle faces (F, 3)

    Returns:
        Edge length variance
    """
    # Get all edges (including duplicates)
    edges = []
    for i in range(3):
        j = (i + 1) % 3
        edges.append(faces[:, [i, j]])

    edges = torch.cat(edges, dim=0)

    # Compute edge lengths
    v0 = vertices[edges[:, 0]]
    v1 = vertices[edges[:, 1]]
    edge_lengths = (v1 - v0).norm(dim=1)

    # Return variance
    return edge_lengths.var().item()


def create_optimization_video(output_dir: Path,
                              test_case_dir: Path,
                              fps: int = 10,
                              device: str = 'cpu'):
    """
    Create video showing mesh evolution during optimization.

    This loads all checkpoint meshes and creates an animation
    showing how the mesh deforms over time.

    Args:
        output_dir: Output directory with checkpoints
        test_case_dir: Test case directory
        fps: Frames per second
        device: Compute device
    """
    from src.mesh import load_mesh
    from src.camera import load_camera_params, create_camera
    from src.renderer import DifferentiableRenderer

    print("\nCreating optimization video...")

    # Find all checkpoint files
    checkpoint_files = sorted(output_dir.glob("checkpoint_*.obj"))

    if len(checkpoint_files) == 0:
        print("⚠ No checkpoint files found")
        return

    # Also include initial and final
    initial_path = test_case_dir / "initial.obj"
    final_path = output_dir / "final_mesh.obj"

    mesh_files = []
    if initial_path.exists():
        mesh_files.append(initial_path)
    mesh_files.extend(checkpoint_files)
    if final_path.exists():
        mesh_files.append(final_path)

    if len(mesh_files) < 2:
        print("⚠ Not enough meshes for video")
        return

    print(f"  - Found {len(mesh_files)} mesh snapshots")

    # Load first mesh to get faces
    _, faces = load_mesh(mesh_files[0])

    # Setup renderer
    renderer = DifferentiableRenderer(image_size=512, device=device)

    # Create camera (rotating view)
    camera = create_camera(azimuth=45, elevation=30, distance=3.0,
                           image_size=512, device=device)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Load optimization history
    history_path = output_dir / "optimization_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
    else:
        history = None

    def animate(frame_idx):
        """Animation function"""
        # Load mesh for this frame
        vertices, _ = load_mesh(mesh_files[frame_idx])
        vertices = vertices.to(device)
        faces_dev = faces.to(device)

        # Render
        rendered = renderer.render_silhouette(
            vertices, faces_dev, camera, soft=False)

        # Display
        axes[0].clear()
        axes[0].imshow(rendered[0, :, :, 0].cpu().numpy(), cmap='gray')
        axes[0].set_title(f'Mesh Evolution - Step {frame_idx}/{len(mesh_files)-1}',
                          fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Plot loss curve up to this point
        if history:
            axes[1].clear()

            # Map frame to iteration
            if frame_idx == 0:
                current_iter = 0
            elif frame_idx == len(mesh_files) - 1:
                current_iter = len(history['total_loss']) - 1
            else:
                # Interpolate
                total_iters = len(history['total_loss'])
                current_iter = int((frame_idx / len(mesh_files)) * total_iters)

            iters = range(current_iter + 1)
            axes[1].plot(iters, history['total_loss'][:current_iter+1],
                         linewidth=2, color='#2E86AB')
            axes[1].set_xlabel('Iteration', fontsize=12)
            axes[1].set_ylabel('Total Loss', fontsize=12)
            axes[1].set_title('Loss Curve', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_xlim(0, len(history['total_loss']))

            # Set consistent y-limits
            all_losses = history['total_loss']
            axes[1].set_ylim(min(all_losses) * 0.9, max(all_losses) * 1.1)

    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(mesh_files),
                         interval=1000//fps, repeat=True)

    # Save as MP4 (requires ffmpeg)
    output_path = output_dir / "optimization_video.mp4"

    try:
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(str(output_path), writer=writer)
        print(f"✓ Saved video: {output_path}")
    except Exception as e:
        print(f"⚠ Could not save video (ffmpeg required): {e}")

        # Save as GIF instead
        try:
            output_path = output_dir / "optimization_animation.gif"
            anim.save(str(output_path), writer='pillow', fps=fps)
            print(f"✓ Saved GIF animation: {output_path}")
        except Exception as e2:
            print(f"⚠ Could not save GIF: {e2}")

    plt.close()


def create_mesh_comparison_grid(meshes: List[Tuple[torch.Tensor, str]],
                                faces: torch.Tensor,
                                cameras: List,
                                output_path: Path,
                                device: str = 'cpu'):
    """
    Create grid comparing multiple meshes from multiple views.

    Args:
        meshes: List of (vertices, name) tuples
        faces: Triangle faces
        cameras: List of cameras
        output_path: Where to save the image
        device: Compute device
    """
    from src.renderer import DifferentiableRenderer

    num_meshes = len(meshes)
    num_views = len(cameras)

    renderer = DifferentiableRenderer(image_size=256, device=device)

    fig, axes = plt.subplots(num_meshes, num_views,
                             figsize=(num_views * 4, num_meshes * 4))

    if num_meshes == 1:
        axes = axes.reshape(1, -1)
    if num_views == 1:
        axes = axes.reshape(-1, 1)

    for row, (vertices, name) in enumerate(meshes):
        vertices = vertices.to(device)
        faces_dev = faces.to(device)

        for col, camera in enumerate(cameras):
            rendered = renderer.render_silhouette(vertices, faces_dev,
                                                  camera, soft=False)

            axes[row, col].imshow(
                rendered[0, :, :, 0].cpu().numpy(), cmap='gray')
            axes[row, col].axis('off')

            if row == 0:
                axes[row, col].set_title(f'View {col+1}', fontsize=12)

        axes[row, 0].set_ylabel(name, fontsize=12, fontweight='bold',
                                rotation=0, ha='right', va='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved mesh comparison: {output_path}")


# ==================== UNIT TESTS ====================

def test_load_target_images():
    """Test loading target images"""
    test_dir = Path("data/test_cases/test_case_a")

    if not test_dir.exists():
        print("⚠ Test case A not found, skipping test")
        return

    images = load_target_images(test_dir, image_size=128)

    assert len(images) > 0, "Should load at least one image"
    assert images[0].shape == (1, 128, 128, 1), "Image should be [1, H, W, 1]"

    print("✓ test_load_target_images passed")


def test_compute_vertex_normals():
    """Test normal computation"""
    from src.mesh import create_icosphere

    vertices, faces = create_icosphere(subdivisions=1)
    normals = compute_vertex_normals(vertices, faces)

    assert normals.shape == vertices.shape, "Normals shape should match vertices"

    # Check normalization
    norms = normals.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        "Normals should be normalized"

    print("✓ test_compute_vertex_normals passed")


if __name__ == "__main__":
    print("Testing visualization utilities...")
    print()

    test_load_target_images()
    print()

    test_compute_vertex_normals()
    print()

    print("="*60)
    print("All utility tests passed! ✓")
    print("="*60)
