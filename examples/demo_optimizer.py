"""
Demo: Full mesh optimization from images!

This script demonstrates the complete inverse rendering pipeline:
1. Load initial mesh (sphere)
2. Load target images (from test case)
3. Setup cameras
4. Optimize mesh to match targets
5. Visualize results

This brings together ALL the pieces:
mesh + camera + renderer + losses + optimizer = 3D reconstruction!
"""

from src.optimizer import MeshOptimizer
from src.renderer import DifferentiableRenderer
from src.camera import load_camera_params
from src.mesh import load_mesh, save_mesh
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_target_images(test_case_dir: Path, image_size: int = 256) -> list:
    """Load all target images from test case"""
    targets_dir = test_case_dir / "targets"
    target_files = sorted(targets_dir.glob("view_*.png"))

    images = []
    for img_file in target_files:
        # Load image
        img = Image.open(img_file).convert('L')  # Grayscale
        img = img.resize((image_size, image_size))

        # Convert to tensor [1, H, W, 1]
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(-1)
        images.append(img_tensor)

    return images


def visualize_results(test_case_dir: Path, output_dir: Path):
    """Create comparison visualization"""
    from src.mesh import load_mesh
    from src.camera import load_camera_params

    # Load meshes
    initial_verts, faces = load_mesh(test_case_dir / "initial.obj")
    gt_verts, _ = load_mesh(test_case_dir / "ground_truth.obj")
    final_verts, _ = load_mesh(output_dir / "final_mesh.obj")

    # Load cameras for rendering
    cameras = load_camera_params(test_case_dir / "camera_config.json")

    # Default azimuths for labels
    azimuths = [0, 90, 180, 270][:len(cameras)]    # Render all three meshes
    renderer = DifferentiableRenderer(image_size=256, device='cpu')

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Mesh Optimization Results', fontsize=16, fontweight='bold')

    meshes = [
        (initial_verts, "Initial (Sphere)"),
        (gt_verts, "Ground Truth"),
        (final_verts, "Optimized")
    ]

    for row, (verts, title) in enumerate(meshes):
        verts = verts.to('cpu')
        faces_cpu = faces.to('cpu')

        for col in range(4):
            camera = cameras[col]
            rendered = renderer.render_silhouette(
                verts, faces_cpu, camera, soft=False
            )

            axes[row, col].imshow(
                rendered[0, :, :, 0].cpu().numpy(), cmap='gray')
            axes[row, col].axis('off')

            if row == 0:
                azim = azimuths[col]
                axes[row, col].set_title(
                    f'View {col+1} (az={azim}°)', fontsize=10)

        axes[row, 0].set_ylabel(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / "optimization_comparison.png",
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison: {output_dir / 'optimization_comparison.png'}")


def run_optimization_demo(test_case: str = 'a',
                          num_iterations: int = 200,
                          learning_rate: float = 0.01,
                          device: str = 'cpu'):
    """
    Run complete optimization demo on a test case.

    Args:
        test_case: 'a', 'b', or 'c'
        num_iterations: Number of optimization steps
        learning_rate: Optimization learning rate
        device: 'cpu' or 'cuda'
    """
    print("="*70)
    print("MESH OPTIMIZATION DEMO")
    print("="*70)

    # Setup paths
    base_dir = Path(__file__).parent
    test_case_dir = base_dir / "test" / f"test_case_{test_case}"
    output_dir = base_dir / "results" / f"optimization_{test_case}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTest case: {test_case}")
    print(f"Output directory: {output_dir}")

    # Load metadata
    with open(test_case_dir / "metadata.json") as f:
        metadata = json.load(f)

    print(f"\nTest case info:")
    print(f"  - Type: {metadata['deformation']}")
    print(f"  - Num views: {metadata['num_views']}")
    print(f"  - Image size: {metadata['image_size']}x{metadata['image_size']}")

    # Load initial mesh
    print("\nLoading initial mesh...")
    initial_vertices, faces = load_mesh(test_case_dir / "initial.obj")
    print(f"  Vertices: {initial_vertices.shape}")
    print(f"  Faces: {faces.shape}")

    # Load targets
    print("\nLoading target images...")
    target_images = load_target_images(
        test_case_dir, image_size=metadata['image_size'])
    print(f"  Loaded {len(target_images)} target images")

    # Load cameras
    print("\nLoading cameras...")
    cameras = load_camera_params(test_case_dir / "camera_config.json")
    print(f"  Loaded {len(cameras)} cameras")

    # Load camera config to get azimuths for visualization
    with open(test_case_dir / "camera_config.json") as f:
        camera_config = json.load(f)
    # Default azimuths for visualization
    azimuths = [0, 90, 180, 270][:len(cameras)]

    # Create renderer
    print("\nInitializing renderer...")
    renderer = DifferentiableRenderer(
        image_size=metadata['image_size'],
        device=device
    )

    # Create optimizer
    print("\nInitializing optimizer...")
    loss_weights = {
        'silhouette': 1.0,
        'edge': 0.1,
        'laplacian': 0.1,
        'normal': 0.01
    }

    optimizer = MeshOptimizer(
        renderer=renderer,
        loss_weights=loss_weights,
        learning_rate=learning_rate,
        optimizer_type='adam',
        device=device
    )

    # Run optimization
    print("\nStarting optimization...")
    optimized_vertices, history = optimizer.optimize_mesh(
        initial_vertices=initial_vertices,
        faces=faces,
        target_images=target_images,
        cameras=cameras,
        num_iterations=num_iterations,
        log_interval=20,
        checkpoint_interval=50,
        output_dir=str(output_dir)
    )

    # Create comparison visualization
    print("\nCreating comparison visualization...")
    visualize_results(test_case_dir, output_dir)

    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print("\nFiles created:")
    print(f"  - final_mesh.obj           : Optimized mesh")
    print(f"  - loss_curve.png          : Loss over iterations")
    print(f"  - optimization_history.json: Detailed loss history")
    print(f"  - optimization_comparison.png: Before/after comparison")
    print(f"  - checkpoint_*.obj        : Intermediate meshes")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Demo mesh optimization')
    parser.add_argument('--test-case', type=str, default='a',
                        choices=['a', 'b', 'c'],
                        help='Test case to run (a=easy, b=medium, c=hard)')
    parser.add_argument('--iterations', type=int, default=200,
                        help='Number of optimization iterations')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')

    args = parser.parse_args()

    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    run_optimization_demo(
        test_case=args.test_case,
        num_iterations=args.iterations,
        learning_rate=args.lr,
        device=args.device
    )
