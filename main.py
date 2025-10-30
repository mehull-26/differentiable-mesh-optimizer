"""
Main Script for 3D Model Reconstruction

This is the main entry point for running the complete pipeline:
1. Load configuration from YAML
2. Setup experiment
3. Load data (initial mesh, targets, cameras)
4. Run optimization
5. Evaluate results
6. Generate visualizations

Usage:
    python main.py --config config.yaml
    python main.py --config config.yaml --experiment my_experiment
    python main.py --config experiments/high_res.yaml
"""

import torch
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import json
import shutil
from src.mesh import create_icosphere, load_mesh, save_mesh
from src.camera import load_camera_params
from src.renderer import DifferentiableRenderer
from src.optimizer import MeshOptimizer
from src.utils import (
    load_target_images,
    visualize_optimization,
    compare_meshes,
    compute_3d_vertex_matching,
    create_optimization_video,
    save_experiment_config,
    setup_experiment_directory
)
import io
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Fix Windows console encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"✓ Loaded configuration from: {config_path}")
    return config


def setup_device(config: dict) -> str:
    """
    Setup compute device (CPU or CUDA).

    Args:
        config: Configuration dictionary

    Returns:
        Device string ('cpu' or 'cuda')
    """
    device_type = config['device']['type']

    if device_type == 'auto':
        if torch.cuda.is_available():
            device = f"cuda:{config['device']['cuda_device']}"
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print("✓ Using CPU (CUDA not available)")
    elif device_type == 'cuda':
        if torch.cuda.is_available():
            device = f"cuda:{config['device']['cuda_device']}"
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA requested but not available, falling back to CPU")
            device = 'cpu'
    else:
        device = 'cpu'
        print("✓ Using CPU")

    return device


def load_data(config: dict, device: str, target_type: str = 'soft') -> tuple:
    """
    Load all data needed for optimization.

    Args:
        config: Configuration dictionary
        device: Compute device
        target_type: Type of targets to load ('soft' or 'hard')

    Returns:
        Tuple of (initial_vertices, faces, target_images, cameras, test_case_dir)
    """
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)

    # Test case directory
    test_case_name = config['test_case']
    test_case_dir = Path("data") / "test_cases" / test_case_name

    if not test_case_dir.exists():
        raise ValueError(f"Test case not found: {test_case_dir}")

    print(f"\nTest case: {test_case_name}")
    print(f"Target type: {target_type}")

    # Load metadata
    with open(test_case_dir / "metadata.json") as f:
        metadata = json.load(f)

    print(f"  - Type: {metadata['deformation']}")
    print(f"  - Num views: {metadata['num_views']}")
    print(f"  - Image size: {metadata['image_size']}x{metadata['image_size']}")

    # Load or create initial mesh
    mesh_config = config['mesh']

    if mesh_config['initial_type'] == 'sphere':
        subdivision_level = mesh_config['subdivision_level']
        print(f"\nCreating initial sphere...")
        print(f"  - Subdivision level: {subdivision_level}")
        vertices, faces = create_icosphere(subdivisions=subdivision_level)
        print(f"  - Vertices: {vertices.shape[0]}")
        print(f"  - Faces: {faces.shape[0]}")

        # Show subdivision level reference
        expected_verts = {0: 12, 1: 42, 2: 162, 3: 642, 4: 2562, 5: 10242}
        if subdivision_level in expected_verts:
            print(
                f"  - Expected for level {subdivision_level}: ~{expected_verts[subdivision_level]} vertices ✓")
    elif (test_case_dir / "initial.obj").exists():
        print(f"\nLoading initial mesh from test case...")
        vertices, faces = load_mesh(test_case_dir / "initial.obj")
        print(f"  - Vertices: {vertices.shape[0]}")
        print(f"  - Faces: {faces.shape[0]}")
    else:
        raise ValueError("No initial mesh found")

    # Load target images from appropriate directory
    print(f"\nLoading target images...")
    target_images = load_target_images(
        test_case_dir,
        image_size=config['rendering']['image_size'],
        target_type=target_type  # NEW: Pass target type
    )
    print(f"  - Loaded {len(target_images)} {target_type} target images")

    # Load cameras
    print(f"\nLoading cameras...")
    cameras = load_camera_params(
        test_case_dir / "camera_config.json", device=device)
    print(f"  - Loaded {len(cameras)} cameras")

    return vertices, faces, target_images, cameras, test_case_dir


def run_optimization(config: dict,
                     vertices: torch.Tensor,
                     faces: torch.Tensor,
                     target_images: list,
                     cameras: list,
                     test_case_dir: Path,
                     output_dir: Path,
                     device: str) -> tuple:
    """
    Run the optimization loop.

    Args:
        config: Configuration dictionary
        vertices: Initial vertices
        faces: Triangle faces
        target_images: List of target images
        cameras: List of cameras
        test_case_dir: Path to test case directory
        output_dir: Output directory
        device: Compute device
        output_dir: Output directory
        device: Compute device

    Returns:
        Tuple of (optimized_vertices, history)
    """
    print("\n" + "="*70)
    print("OPTIMIZATION")
    print("="*70)

    # Create renderer
    render_config = config['rendering']
    renderer = DifferentiableRenderer(
        image_size=render_config['image_size'],
        blur_radius=render_config['blur_radius'],
        device=device
    )

    # Create optimizer
    opt_config = config['optimization']
    loss_config = config['losses']

    loss_weights = {
        'silhouette': loss_config['silhouette_weight'],
        'edge': loss_config['edge_weight'],
        'laplacian': loss_config['laplacian_weight'],
        'normal': loss_config['normal_weight']
    }

    optimizer = MeshOptimizer(
        renderer=renderer,
        loss_weights=loss_weights,
        learning_rate=opt_config['learning_rate'],
        optimizer_type=opt_config['optimizer'],
        adaptive_regularization=loss_config.get(
            'adaptive_regularization', False),
        device=device
    )

    # Save initial mesh as checkpoint with consistent naming: checkpoint_0000_00
    from src.mesh import save_mesh, load_mesh
    checkpoint_mesh_path = output_dir / "checkpoint_0000_00.obj"
    save_mesh(vertices.cpu(), faces.cpu(), str(checkpoint_mesh_path))
    print(f"✓ Saved initial mesh: {checkpoint_mesh_path}")

    # Save ground truth reference with same subdivision level
    gt_path = test_case_dir / "ground_truth.obj"
    if gt_path.exists():
        # Load and recreate GT with same subdivision level
        from src.mesh import create_deformed_sphere

        # Get deformation type from metadata
        with open(test_case_dir / "metadata.json") as f:
            metadata = json.load(f)

        deformation = metadata.get('deformation', {})
        deform_type = deformation.get('type', 'ellipsoid')
        deform_params = deformation.get('params', {})

        # Create GT with same subdivision as optimized mesh
        subdivision_level = config['mesh']['subdivision_level']
        gt_vertices_highres = create_deformed_sphere(
            deformation_type=deform_type,
            params=deform_params,
            subdivisions=subdivision_level
        )
        gt_vertices, gt_faces = gt_vertices_highres

        gt_mesh_path = output_dir / "reference_ground_truth.obj"
        save_mesh(gt_vertices, gt_faces, str(gt_mesh_path))
        print(f"✓ Saved ground truth reference: {gt_mesh_path}")
        print(f"   - Subdivision level: {subdivision_level}")
        print(
            f"   - Vertices: {gt_vertices.shape[0]}, Faces: {gt_faces.shape[0]}")

        # Save ACTUAL target images as ground truth reference
        vis_config = config['visualization']
        if vis_config.get('save_checkpoint_renders', True):
            print(f"✓ Saving ground truth (actual target images)...")
            renders_dir = output_dir / "checkpoint_renders"
            renders_dir.mkdir(exist_ok=True)

            num_target_images = len(target_images)

            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(
                1, num_target_images, figsize=(4*num_target_images, 4))
            if num_target_images == 1:
                axes = [axes]
            fig.suptitle('Ground Truth Target Images',
                         fontsize=16, fontweight='bold')

            for idx, target_img in enumerate(target_images):
                # Display actual target image
                axes[idx].imshow(
                    target_img[0, :, :, 0].detach().cpu().numpy(), cmap='gray')
                axes[idx].set_title(f'View {idx}', fontsize=12)
                axes[idx].axis('off')

            plt.tight_layout()
            output_path = renders_dir / "reference_ground_truth.png"
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved ground truth target images: {output_path}")

    # Save initial mesh renders to match target image viewpoints
    vis_config = config['visualization']
    if vis_config.get('save_checkpoint_renders', True):
        print(f"✓ Saving initial mesh renders...")
        renders_dir = output_dir / "checkpoint_renders"
        renders_dir.mkdir(exist_ok=True)

        num_target_images = len(target_images)

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, num_target_images,
                                 figsize=(4*num_target_images, 4))
        if num_target_images == 1:
            axes = [axes]
        fig.suptitle('Iteration 0 (0% - Initial Mesh)',
                     fontsize=16, fontweight='bold')

        # Render from the SAME viewpoints as the target images
        for idx, camera in enumerate(cameras):
            rendered = renderer.render_silhouette(
                vertices, faces, camera
            )
            axes[idx].imshow(
                rendered[0, :, :, 0].detach().cpu().numpy(), cmap='gray')
            axes[idx].set_title(f'View {idx}', fontsize=12)
            axes[idx].axis('off')

        plt.tight_layout()
        output_path = renders_dir / "checkpoint_0000_00.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved initial renders: {output_path}")

    # Run optimization

    optimized_vertices, history = optimizer.optimize_mesh(
        initial_vertices=vertices,
        faces=faces,
        target_images=target_images,
        cameras=cameras,
        num_iterations=opt_config['num_iterations'],
        log_interval=vis_config['log_every'],
        checkpoint_interval=vis_config['save_checkpoint_every'],
        checkpoint_mode=vis_config.get('checkpoint_mode', 'iteration'),
        save_checkpoint_renders=vis_config.get(
            'save_checkpoint_renders', True),
        checkpoint_render_views=vis_config.get('checkpoint_render_views', 4),
        verbose=vis_config.get('verbose', True),
        output_dir=str(output_dir)
    )

    return optimized_vertices, history


def evaluate_results(config: dict,
                     optimized_vertices: torch.Tensor,
                     faces: torch.Tensor,
                     test_case_dir: Path,
                     output_dir: Path) -> dict:
    """
    Evaluate optimization results against ground truth.

    Args:
        config: Configuration dictionary
        optimized_vertices: Optimized mesh vertices
        faces: Triangle faces
        test_case_dir: Test case directory
        output_dir: Output directory

    Returns:
        Dictionary of evaluation metrics
    """
    eval_config = config['evaluation']

    if not eval_config.get('compute_chamfer', False):
        return {}

    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)

    # Load ground truth
    gt_path = test_case_dir / "ground_truth.obj"
    if not gt_path.exists():
        print("⚠ No ground truth mesh found, skipping evaluation")
        return {}

    gt_vertices, gt_faces = load_mesh(gt_path)

    # Compare meshes (each with their own faces)
    metrics = compare_meshes(
        optimized_vertices.cpu(),
        gt_vertices,
        faces.cpu(),
        gt_faces=gt_faces.cpu(),  # Pass ground truth faces separately
        compute_chamfer=eval_config.get('compute_chamfer', True),
        compute_normal_consistency=eval_config.get(
            'compute_normal_consistency', True),
        compute_edge_variance=eval_config.get(
            'compute_edge_length_variance', True)
    )

    # Add 3D vertex matching metric
    if optimized_vertices.shape[0] == gt_vertices.shape[0]:
        print("\n" + "="*70)
        vertex_matching = compute_3d_vertex_matching(
            optimized_vertices.cpu(),
            gt_vertices,
            alignment='procrustes'
        )
        metrics.update(vertex_matching)
        print("="*70)
    else:
        print(f"\n⚠ Skipping 3D vertex matching (different vertex counts)")

    # Save evaluation report
    if eval_config.get('save_evaluation_report', True):
        report_path = output_dir / "evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Saved evaluation report: {report_path}")

    return metrics


def generate_visualizations(config: dict,
                            history: dict,
                            test_case_dir: Path,
                            output_dir: Path,
                            device: str):
    """
    Generate all visualizations.

    Args:
        config: Configuration dictionary
        history: Optimization history
        test_case_dir: Test case directory
        output_dir: Output directory
        device: Compute device
    """
    vis_config = config['visualization']

    print("\n" + "="*70)
    print("VISUALIZATION")
    print("="*70)

    # Visualize optimization progress
    if vis_config.get('plot_loss_curves', True):
        print("\nCreating loss curves...")
        visualize_optimization(history, output_dir)

    # Create optimization video
    if vis_config.get('save_video', True):
        print("\nCreating optimization video...")
        try:
            create_optimization_video(
                output_dir,
                test_case_dir,
                fps=vis_config.get('video_fps', 10),
                device=device
            )
        except Exception as e:
            print(f"⚠ Could not create video: {e}")

    print("\n✓ Visualization complete")


def check_subdivision_match(config: dict, test_case_dir: Path) -> bool:
    """
    Check if config subdivision level matches test case subdivision level.
    Prompts user to regenerate if mismatch detected.

    Returns:
        True if match or user wants to continue, False if user wants to exit
    """
    metadata_path = test_case_dir / "metadata.json"
    if not metadata_path.exists():
        print("⚠ Warning: metadata.json not found, cannot verify subdivision level")
        return True

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    test_case_subdivision = metadata.get('subdivision_level')
    config_subdivision = config['mesh']['subdivision_level']

    if test_case_subdivision is None:
        print("⚠ Warning: Test case doesn't specify subdivision_level in metadata")
        return True

    if test_case_subdivision != config_subdivision:
        print("\n" + "⚠"*35)
        print("SUBDIVISION MISMATCH DETECTED")
        print("⚠"*35)
        print(f"  Config: subdivision_level = {config_subdivision}")

        # Calculate vertex counts
        vertex_counts = [12, 42, 162, 642, 2562, 10242, 40962]
        config_vertices = vertex_counts[config_subdivision] if config_subdivision < len(
            vertex_counts) else "Unknown"
        test_vertices = vertex_counts[test_case_subdivision] if test_case_subdivision < len(
            vertex_counts) else "Unknown"

        print(f"    → {config_vertices} vertices")
        print(
            f"  Test case '{test_case_dir.name}': subdivision_level = {test_case_subdivision}")
        print(f"    → {test_vertices} vertices")
        print()
        print("This mismatch may affect evaluation accuracy.")
        print()
        print("Options:")
        print(
            "  1. Regenerate test case with subdivision_level={config_subdivision}")
        print("  2. Continue anyway (not recommended)")
        print("  3. Exit and update config")

        while True:
            choice = input("\nEnter choice (1/2/3): ").strip()
            if choice == '1':
                print("\nRegenerating test case...")
                return 'regenerate'
            elif choice == '2':
                print("\n⚠ Continuing with mismatched subdivision levels...")
                return True
            elif choice == '3':
                print("\nExiting. Please update your config and try again.")
                return False
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

    return True


def check_target_availability(config: dict, test_case_dir: Path, target_type: str) -> bool:
    """
    Check if requested target type exists. Prompts user to generate if missing.

    Args:
        config: Configuration dictionary
        test_case_dir: Path to test case directory
        target_type: "soft" or "hard"

    Returns:
        True if targets exist or were generated, False if user wants to exit
    """
    targets_dir = test_case_dir / f"targets_{target_type}"

    if not targets_dir.exists() or len(list(targets_dir.glob("view_*.png"))) == 0:
        print("\n" + "⚠"*35)
        print(f"MISSING {target_type.upper()} TARGETS")
        print("⚠"*35)
        print(
            f"  Config requests: target_rendering_mode = '{target_type}' (or 'both')")
        print(f"  But {targets_dir} doesn't exist or is empty")
        print()
        print("Options:")
        print(f"  1. Generate {target_type} targets now")
        print("  2. Skip this target type")
        print("  3. Exit")

        while True:
            choice = input("\nEnter choice (1/2/3): ").strip()
            if choice == '1':
                print(f"\nGenerating {target_type} targets...")
                return 'generate'
            elif choice == '2':
                print(f"\n⚠ Skipping {target_type} target optimization...")
                return 'skip'
            elif choice == '3':
                print("\nExiting.")
                return False
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

    return True


def main(config_path: str, experiment_name: str = None):
    """
    Main pipeline execution with support for dual rendering modes.

    Supports running optimization with:
    - soft targets only
    - hard targets only  
    - both (runs twice, once for each)

    Args:
        config_path: Path to configuration YAML
        experiment_name: Optional experiment name
    """
    print("="*70)
    print("3D MODEL RECONSTRUCTION - MAIN PIPELINE")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load configuration
    config = load_config(config_path)

    # Setup base experiment directory
    if experiment_name:
        config['output']['experiment_name'] = experiment_name

    base_output_dir = Path(config['output']['base_dir']) / \
        config['output']['experiment_name']
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n✓ Base experiment directory: {base_output_dir}")

    # Setup device
    device = setup_device(config)

    # Set random seed for reproducibility
    if 'random_seed' in config.get('advanced', {}):
        seed = config['advanced']['random_seed']
        torch.manual_seed(seed)
        print(f"\n✓ Random seed: {seed}")

    # Determine test case directory
    test_case_name = config['test_case']
    test_case_dir = Path(f"data/test_cases/{test_case_name}")

    # Check subdivision level match
    subdivision_check = check_subdivision_match(config, test_case_dir)
    if subdivision_check == 'regenerate':
        # Regenerate test case with matching subdivision
        print("\nCalling test case generation...")
        # Update generation config temporarily
        gen_config_path = Path("data/generation_config.yaml")
        with open(gen_config_path, 'r', encoding='utf-8') as f:
            gen_config = yaml.safe_load(f)

        # Update subdivision in generation config
        original_subdivision = gen_config['mesh']['subdivision_level']
        gen_config['mesh']['subdivision_level'] = config['mesh']['subdivision_level']

        with open(gen_config_path, 'w') as f:
            yaml.dump(gen_config, f, default_flow_style=False)

        # Run generation
        import subprocess
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path.cwd())
        env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        result = subprocess.run(
            ['python', 'data/generate_targets.py'],
            input='1\n',
            text=True,
            env=env,
            capture_output=False
        )

        if result.returncode != 0:
            print("\n✗ Test case generation failed")
            return

        print("\n✓ Test case regenerated successfully")

    elif subdivision_check == False:
        print("\nExiting...")
        return

    # Determine which target types to use
    target_rendering_mode = config['rendering'].get(
        'target_rendering_mode', 'soft')

    if target_rendering_mode == 'both':
        target_types = ['soft', 'hard']
    else:
        target_types = [target_rendering_mode]

    print(f"\n✓ Target rendering mode: {target_rendering_mode}")
    print(f"  Will run optimization for: {', '.join(target_types)}")

    # Run optimization for each target type
    for target_type in target_types:
        print("\n" + "="*70)
        print(f"OPTIMIZATION WITH {target_type.upper()} TARGETS")
        print("="*70)

        # Check target availability
        availability = check_target_availability(
            config, test_case_dir, target_type)

        if availability == 'generate':
            # Generate missing targets
            print(f"\nGenerating {target_type} targets...")
            # Update generation config to generate only this type
            gen_config_path = Path("data/generation_config.yaml")
            with open(gen_config_path, 'r') as f:
                gen_config = yaml.safe_load(f)

            gen_config['rendering']['target_types'] = [target_type]

            with open(gen_config_path, 'w') as f:
                yaml.dump(gen_config, f, default_flow_style=False)

            # Run generation for specific test case
            import subprocess
            env = os.environ.copy()
            env['PYTHONPATH'] = str(Path.cwd())
            env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            result = subprocess.run(
                ['python', 'data/generate_targets.py'],
                input='1\n',
                text=True,
                env=env,
                capture_output=False
            )

            if result.returncode != 0:
                print(
                    f"\n✗ Failed to generate {target_type} targets, skipping...")
                continue

        elif availability == 'skip':
            continue
        elif availability == False:
            print("\nExiting...")
            return

        # Create output directory for this target type
        output_dir = base_output_dir / f"{target_type}_target"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n✓ Output directory: {output_dir}")

        # Save configuration to output
        save_experiment_config(config, output_dir)

        # Load data with specific target type
        vertices, faces, target_images, cameras, test_case_dir = load_data(
            config, device, target_type=target_type
        )

        # Run optimization
        optimized_vertices, history = run_optimization(
            config, vertices, faces, target_images, cameras, test_case_dir, output_dir, device
        )

        # Evaluate results
        metrics = evaluate_results(
            config, optimized_vertices, faces, test_case_dir, output_dir
        )

        # Generate visualizations
        generate_visualizations(
            config, history, test_case_dir, output_dir, device
        )

        print(f"\n✓ {target_type.upper()} target optimization complete!")
        print(f"  Results: {output_dir}")

    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {base_output_dir}")

    if len(target_types) > 1:
        print("\nCompleted optimizations:")
        for target_type in target_types:
            print(f"  • {target_type}_target/")

    print("\nGenerated files (in each target folder):")
    print("  Meshes:")
    print("    - checkpoint_0000_00.obj       : Initial mesh")
    print("    - checkpoint_NNNN_PP.obj       : Intermediate checkpoints")
    print("    - checkpoint_FFFF_100.obj      : Final optimized mesh")
    print("    - reference_ground_truth.obj   : Ground truth reference")
    print("  Renders:")
    print("    - checkpoint_renders/checkpoint_0000_00.png : Initial state")
    print("    - checkpoint_renders/checkpoint_NNNN_PP.png : Progress views")
    print("    - checkpoint_renders/checkpoint_FFFF_100.png : Final result")
    print("    - checkpoint_renders/reference_ground_truth.png : GT reference")
    print("  Metrics:")
    print("    - loss_curve.png               : Loss curves")
    print("    - loss_curves_detailed.png     : Detailed loss breakdown")
    print("    - optimization_history.json    : Training metrics")
    print("    - config.yaml                  : Experiment configuration")
    print("    - evaluation_report.json       : Evaluation metrics")

    print("\n" + "="*70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print("    - config.yaml                  : Experiment configuration")

    if metrics:
        print("    - evaluation_report.json       : Evaluation metrics")

    if config['visualization'].get('save_video', False):
        print("  Animation:")
        print("    - optimization_video.mp4       : Optimization animation")

    print("\n" + "="*70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='3D Model Reconstruction via Inverse Rendering'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default=None,
        help='Experiment name (auto-generated if not provided)'
    )

    args = parser.parse_args()

    # Check if config exists
    if not Path(args.config).exists():
        print(f"Error: Configuration file not found: {args.config}")
        print("\nCreating default config.yaml...")
        # Could copy default config here
        sys.exit(1)

    main(args.config, args.experiment)
