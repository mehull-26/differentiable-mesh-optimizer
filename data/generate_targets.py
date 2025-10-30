"""
Target Generation Script for 3D Reconstruction Test Cases

This script creates synthetic "ground truth" data for testing our 3D
reconstruction pipeline. We'll render known 3D shapes from multiple views,
then later try to reconstruct them from those images.

Why create synthetic data?
- Know the exact ground truth (can measure reconstruction error)
- Control difficulty (simple shapes → complex shapes)
- No need to collect real images initially
- Fast iteration for learning

Test Cases:
- Test Case A: Sphere → Ellipsoid (stretch deformation)
- Test Case B: Sphere → Bump (local deformation)
- Test Case C: Sphere → Complex shape (multiple deformations)
"""

from src.renderer import DifferentiableRenderer, ClassicRenderer, save_rendered_image
from src.camera import create_camera, save_camera_params
from src.mesh import create_icosphere, create_deformed_sphere, save_mesh
import yaml
from pathlib import Path
import json
import numpy as np
import torch
import sys
import os

# Add project root to path BEFORE importing src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_generation_config(config_path: str = "generation_config.yaml"):
    """Load the generation configuration from YAML file."""
    # If absolute or already has data/ prefix, use as is
    # Otherwise, look in the same directory as this script
    config_path_obj = Path(config_path)
    if not config_path_obj.is_absolute() and not str(config_path).startswith('data'):
        script_dir = Path(__file__).parent
        config_path_obj = script_dir / config_path

    with open(config_path_obj, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def generate_test_case_a():
    """
    Test Case A: Sphere to Ellipsoid

    Difficulty: EASY
    - Ground truth: Ellipsoid (uniformly stretched sphere)
    - Initial guess: Regular sphere
    - Task: Learn the stretching along each axis
    - Views: 4 angles around the object (every 90 degrees)

    What makes this easy?
    - Smooth, global deformation
    - Symmetric shape
    - Multiple views show the stretching clearly
    """
    print("=" * 70)
    print("Generating Test Case A: Sphere → Ellipsoid")
    print("=" * 70)

    # Setup directories
    output_dir = Path("data/test_cases/test_case_a")
    targets_dir = output_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    # Parameters
    image_size = 256
    num_views = 4
    elevation = 30.0  # Look down at 30 degrees
    distance = 3.0    # Camera distance
    device = 'cpu'    # Use 'cuda' for faster rendering if available

    # 1. Create ground truth mesh (ellipsoid)
    print("\n[1/5] Creating ground truth mesh...")
    gt_vertices, gt_faces = create_deformed_sphere(
        deformation_type='ellipsoid',
        params={'scale': [1.5, 1.0, 0.8]}  # Stretch X, normal Y, compress Z
    )

    # Save ground truth mesh
    gt_mesh_path = output_dir / "ground_truth.obj"
    save_mesh(gt_vertices, gt_faces, str(gt_mesh_path))
    print(f"  Saved ground truth mesh: {gt_mesh_path}")
    print(f"  Vertices: {gt_vertices.shape[0]}, Faces: {gt_faces.shape[0]}")

    # 2. Create initial mesh (regular sphere)
    print("\n[2/5] Creating initial mesh (sphere)...")
    initial_vertices, initial_faces = create_icosphere(subdivisions=2)
    initial_mesh_path = output_dir / "initial.obj"
    save_mesh(initial_vertices, initial_faces, str(initial_mesh_path))
    print(f"  Saved initial mesh: {initial_mesh_path}")

    # 3. Define camera positions
    print("\n[3/5] Setting up cameras...")
    cameras = []
    camera_configs = []

    for i in range(num_views):
        azimuth = (360.0 / num_views) * i
        camera = create_camera(
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
            image_size=image_size,
            device=device
        )
        cameras.append(camera)
        camera_configs.append({
            'azimuth': azimuth,
            'elevation': elevation,
            'distance': distance,
            'image_size': image_size
        })
        print(
            f"  Camera {i}: azimuth={azimuth:.0f}°, elevation={elevation:.0f}°")

    # Save camera configuration
    camera_config_path = output_dir / "camera_config.json"
    save_camera_params(
        cameras,
        str(camera_config_path),
        metadata={
            'num_views': num_views,
            'elevation': elevation,
            'distance': distance,
            'image_size': image_size,
            'description': 'Test Case A: Sphere to Ellipsoid'
        }
    )
    print(f"  Saved camera config: {camera_config_path}")

    # 4. Render target images from ground truth
    print("\n[4/5] Rendering target images...")
    renderer = DifferentiableRenderer(image_size=image_size, device=device)

    target_paths = []
    for i, camera in enumerate(cameras):
        # Render silhouette
        silhouette = renderer.render_silhouette(
            gt_vertices,
            gt_faces,
            camera,
            soft=True
        )

        # Save target image
        target_path = targets_dir / f"view_{i:02d}.png"
        save_rendered_image(silhouette, str(target_path))
        target_paths.append(str(target_path))

        azimuth = camera_configs[i]['azimuth']
        print(f"  Rendered view {i} (azimuth={azimuth:.0f}°): {target_path}")

    # 5. Create test case metadata
    print("\n[5/5] Saving metadata...")
    metadata = {
        'test_case': 'A',
        'name': 'Sphere to Ellipsoid',
        'difficulty': 'easy',
        'description': 'Uniformly stretched sphere along X and Z axes',
        'ground_truth_mesh': str(gt_mesh_path),
        'initial_mesh': str(initial_mesh_path),
        'camera_config': str(camera_config_path),
        'num_views': num_views,
        'image_size': image_size,
        'deformation': {
            'type': 'ellipsoid',
            'params': {'scale': [1.5, 1.0, 0.8]}
        },
        'target_images': target_paths,
        'expected_vertices': gt_vertices.shape[0],
        'expected_faces': gt_faces.shape[0]
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved metadata: {metadata_path}")

    print("\n" + "=" * 70)
    print("✓ Test Case A generated successfully!")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Target images: {targets_dir}")
    print(f"\nNext steps:")
    print(f"  1. Examine the target images in {targets_dir}")
    print(f"  2. Load the meshes in MeshLab/Blender to visualize")
    print(f"  3. Use this test case for optimization!")
    print()


def generate_test_case_b():
    """
    Test Case B: Sphere to Bump

    Difficulty: MEDIUM
    - Ground truth: Sphere with a local bump on one side
    - Initial guess: Regular sphere
    - Task: Learn the local deformation
    - Views: 8 angles (need more views to see the bump from different angles)

    What makes this harder than A?
    - Local deformation (not uniform)
    - Asymmetric shape
    - Requires more views to capture the bump
    """
    print("=" * 70)
    print("Generating Test Case B: Sphere → Bump")
    print("=" * 70)

    output_dir = Path("data/test_cases/test_case_b")
    targets_dir = output_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    image_size = 256
    num_views = 8  # More views for asymmetric shape
    elevation = 30.0
    distance = 3.0
    device = 'cpu'

    # 1. Create ground truth with bump
    print("\n[1/5] Creating ground truth mesh...")
    gt_vertices, gt_faces = create_deformed_sphere(
        deformation_type='bump',
        params={'amplitude': 0.4}  # Bump size
    )

    gt_mesh_path = output_dir / "ground_truth.obj"
    save_mesh(gt_vertices, gt_faces, str(gt_mesh_path))
    print(f"  Saved ground truth mesh: {gt_mesh_path}")

    # 2. Create initial mesh
    print("\n[2/5] Creating initial mesh...")
    initial_vertices, initial_faces = create_icosphere(subdivisions=2)
    initial_mesh_path = output_dir / "initial.obj"
    save_mesh(initial_vertices, initial_faces, str(initial_mesh_path))
    print(f"  Saved initial mesh: {initial_mesh_path}")

    # 3. Setup cameras
    print("\n[3/5] Setting up cameras...")
    cameras = []
    camera_configs = []

    for i in range(num_views):
        azimuth = (360.0 / num_views) * i
        camera = create_camera(
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
            image_size=image_size,
            device=device
        )
        cameras.append(camera)
        camera_configs.append({
            'azimuth': azimuth,
            'elevation': elevation,
            'distance': distance
        })
        print(f"  Camera {i}: azimuth={azimuth:.0f}°")

    camera_config_path = output_dir / "camera_config.json"
    save_camera_params(
        cameras,
        str(camera_config_path),
        metadata={
            'num_views': num_views,
            'elevation': elevation,
            'distance': distance,
            'image_size': image_size,
            'description': 'Test Case B: Sphere to Bump'
        }
    )

    # 4. Render targets
    print("\n[4/5] Rendering target images...")
    renderer = DifferentiableRenderer(image_size=image_size, device=device)

    target_paths = []
    for i, camera in enumerate(cameras):
        silhouette = renderer.render_silhouette(
            gt_vertices, gt_faces, camera, soft=True
        )

        target_path = targets_dir / f"view_{i:02d}.png"
        save_rendered_image(silhouette, str(target_path))
        target_paths.append(str(target_path))

        print(f"  Rendered view {i}: {target_path}")

    # 5. Save metadata
    print("\n[5/5] Saving metadata...")
    metadata = {
        'test_case': 'B',
        'name': 'Sphere to Bump',
        'difficulty': 'medium',
        'description': 'Sphere with local bump deformation',
        'ground_truth_mesh': str(gt_mesh_path),
        'initial_mesh': str(initial_mesh_path),
        'camera_config': str(camera_config_path),
        'num_views': num_views,
        'image_size': image_size,
        'deformation': {
            'type': 'bump',
            'params': {'amplitude': 0.4}
        },
        'target_images': target_paths
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved metadata: {metadata_path}")

    print("\n" + "=" * 70)
    print("✓ Test Case B generated successfully!")
    print("=" * 70)
    print()


def generate_test_case_c():
    """
    Test Case C: Sphere to Complex Shape

    Difficulty: HARD
    - Ground truth: Multiple deformations combined
    - Initial guess: Regular sphere
    - Task: Learn complex shape
    - Views: 12 angles (comprehensive coverage)

    What makes this hardest?
    - Multiple types of deformations
    - Complex geometry
    - Requires many views
    - Tests optimization robustness
    """
    print("=" * 70)
    print("Generating Test Case C: Sphere → Complex Shape")
    print("=" * 70)

    output_dir = Path("data/test_cases/test_case_c")
    targets_dir = output_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    image_size = 256
    num_views = 12  # Comprehensive coverage
    elevation = 30.0
    distance = 3.0
    device = 'cpu'

    # 1. Create complex ground truth
    print("\n[1/5] Creating ground truth mesh...")
    # Start with ellipsoid
    gt_vertices, gt_faces = create_deformed_sphere(
        deformation_type='ellipsoid',
        params={'scale': [1.3, 0.9, 1.2]}
    )
    # Add bump on top
    gt_vertices_bump, _ = create_deformed_sphere(
        deformation_type='bump',
        params={'amplitude': 0.3}
    )
    # Combine: use ellipsoid base + bump offsets where x > 0
    mask = gt_vertices[:, 0] > 0
    gt_vertices[mask] = gt_vertices[mask] * 0.7 + gt_vertices_bump[mask] * 0.3

    gt_mesh_path = output_dir / "ground_truth.obj"
    save_mesh(gt_vertices, gt_faces, str(gt_mesh_path))
    print(f"  Saved ground truth mesh: {gt_mesh_path}")

    # 2. Create initial mesh
    print("\n[2/5] Creating initial mesh...")
    initial_vertices, initial_faces = create_icosphere(subdivisions=2)
    initial_mesh_path = output_dir / "initial.obj"
    save_mesh(initial_vertices, initial_faces, str(initial_mesh_path))
    print(f"  Saved initial mesh: {initial_mesh_path}")

    # 3. Setup cameras
    print("\n[3/5] Setting up cameras...")
    cameras = []
    camera_configs = []

    for i in range(num_views):
        azimuth = (360.0 / num_views) * i
        camera = create_camera(
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
            image_size=image_size,
            device=device
        )
        cameras.append(camera)
        camera_configs.append({
            'azimuth': azimuth,
            'elevation': elevation,
            'distance': distance
        })
        print(f"  Camera {i}: azimuth={azimuth:.0f}°")

    camera_config_path = output_dir / "camera_config.json"
    save_camera_params(
        cameras,
        str(camera_config_path),
        metadata={
            'num_views': num_views,
            'elevation': elevation,
            'distance': distance,
            'image_size': image_size,
            'description': 'Test Case C: Complex Shape'
        }
    )

    # 4. Render targets
    print("\n[4/5] Rendering target images...")
    renderer = DifferentiableRenderer(image_size=image_size, device=device)

    target_paths = []
    for i, camera in enumerate(cameras):
        silhouette = renderer.render_silhouette(
            gt_vertices, gt_faces, camera, soft=True
        )

        target_path = targets_dir / f"view_{i:02d}.png"
        save_rendered_image(silhouette, str(target_path))
        target_paths.append(str(target_path))

        print(f"  Rendered view {i}: {target_path}")

    # 5. Save metadata
    print("\n[5/5] Saving metadata...")
    metadata = {
        'test_case': 'C',
        'name': 'Complex Shape',
        'difficulty': 'hard',
        'description': 'Combined ellipsoid and bump deformations',
        'ground_truth_mesh': str(gt_mesh_path),
        'initial_mesh': str(initial_mesh_path),
        'camera_config': str(camera_config_path),
        'num_views': num_views,
        'image_size': image_size,
        'deformation': {
            'type': 'combined',
            'params': {
                'ellipsoid_scale': [1.3, 0.9, 1.2],
                'bump_amplitude': 0.3
            }
        },
        'target_images': target_paths
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved metadata: {metadata_path}")

    print("\n" + "=" * 70)
    print("✓ Test Case C generated successfully!")
    print("=" * 70)
    print()


def visualize_test_case(test_case_dir: str):
    """
    Create a visualization showing all target views for a test case.
    """
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        print("matplotlib or PIL not available, skipping visualization")
        return

    test_case_path = Path(test_case_dir)

    # Try to find targets directory (prefer soft targets)
    targets_dir = test_case_path / "targets_soft"
    if not targets_dir.exists():
        targets_dir = test_case_path / "targets_hard"
    if not targets_dir.exists():
        targets_dir = test_case_path / "targets"  # Legacy fallback

    if not targets_dir.exists():
        print(f"⚠ No target images found for visualization")
        return

    # Load metadata
    with open(test_case_path / "metadata.json", 'r') as f:
        metadata = json.load(f)

    # Load all target images
    target_images = []
    for img_path in sorted(targets_dir.glob("view_*.png")):
        img = Image.open(img_path)
        target_images.append(np.array(img))

    if len(target_images) == 0:
        print(f"⚠ No target images found in {targets_dir}")
        return

    # Create grid visualization
    num_images = len(target_images)
    cols = min(4, num_images)
    rows = max(1, (num_images + cols - 1) // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    # Make axes always 2D for consistent indexing
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])
    else:
        axes = np.array(axes)

    for idx, img in enumerate(target_images):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        if img.ndim == 3 and img.shape[2] == 4:
            # RGBA - show only alpha channel
            ax.imshow(img[:, :, 3], cmap='gray')
        else:
            ax.imshow(img, cmap='gray')

        ax.set_title(f"View {idx} (azimuth={(360/num_images)*idx:.0f}°)")
        ax.axis('off')

    # Hide unused subplots
    for idx in range(num_images, rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        ax.axis('off')

    plt.suptitle(f"Test Case {metadata['test_case']}: {metadata['name']}",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save visualization
    viz_path = test_case_path / "visualization.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {viz_path}")
    plt.close()


def generate_from_config(config_path: str = "generation_config.yaml"):
    """
    Generate all test cases using parameters from configuration file.

    This function reads generation_config.yaml and creates test cases
    with configurable number of views, image sizes, and camera positions.
    """
    # Load config
    config = load_generation_config(config_path)

    rendering_config = config['rendering']
    camera_config = config['camera']
    mesh_config = config['mesh']
    test_cases_config = config['test_cases']

    print("=" * 70)
    print("GENERATING TEST CASES FROM CONFIG")
    print("=" * 70)
    print(f"Config file: {config_path}")
    print(f"Number of views: {rendering_config['num_views']}")
    print(f"Image size: {rendering_config['image_size']}")
    print(f"Subdivision level: {mesh_config['subdivision_level']}")
    print()

    # Determine device
    device_type = config.get('device', {}).get('type', 'auto')
    if device_type == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_type

    # Generate each test case
    for test_case_id, test_case_data in test_cases_config.items():
        print(f"\n{'='*70}")
        print(f"Generating {test_case_id}")
        print(f"{'='*70}")
        print(f"Description: {test_case_data['description']}")
        print(f"Difficulty: {test_case_data['difficulty']}")

        # Setup directories
        output_dir = Path(f"data/test_cases/{test_case_id}")
        targets_dir = output_dir / "targets"
        targets_dir.mkdir(parents=True, exist_ok=True)

        # Extract parameters
        image_size = rendering_config['image_size']
        num_views = rendering_config['num_views']
        elevation = camera_config['elevation']
        distance = camera_config['distance']
        subdivision_level = mesh_config['subdivision_level']

        deformation = test_case_data['deformation']
        deform_type = deformation['type']
        deform_params = deformation['params']

        # 1. Create ground truth mesh
        print("\n[1/5] Creating ground truth mesh...")
        gt_vertices, gt_faces = create_deformed_sphere(
            deformation_type=deform_type,
            params=deform_params,
            subdivisions=subdivision_level
        )

        gt_mesh_path = output_dir / "ground_truth.obj"
        save_mesh(gt_vertices, gt_faces, str(gt_mesh_path))
        print(f"  Saved: {gt_mesh_path}")
        print(
            f"  Vertices: {gt_vertices.shape[0]}, Faces: {gt_faces.shape[0]}")

        # 2. Create initial mesh
        print("\n[2/5] Creating initial mesh...")
        initial_vertices, initial_faces = create_icosphere(
            subdivisions=subdivision_level)
        initial_mesh_path = output_dir / "initial.obj"
        save_mesh(initial_vertices, initial_faces, str(initial_mesh_path))
        print(f"  Saved: {initial_mesh_path}")

        # 3. Setup cameras
        print("\n[3/5] Setting up cameras...")
        cameras = []
        camera_configs = []

        # Get elevation configuration
        azimuth_mode = camera_config.get('azimuth_mode', 'uniform')
        elevation_mode = camera_config.get(
            'elevation_mode', 'mixed')  # mixed, fixed, uniform
        elevation_range = camera_config.get('elevation_range', [0, 60])

        # Create varied camera viewpoints for better coverage
        for i in range(num_views):
            # Azimuth (horizontal angle around object)
            if azimuth_mode == 'uniform':
                azimuth = (360.0 / num_views) * i
            else:  # fallback to uniform
                azimuth = (360.0 / num_views) * i

            # Elevation (vertical angle - varies to get top/side/angled views)
            if elevation_mode == 'fixed':
                # Use fixed elevation from config
                elev = elevation
            elif elevation_mode == 'uniform':
                # Distribute elevations uniformly across range
                elev = elevation_range[0] + (elevation_range[1] -
                                             elevation_range[0]) * (i / max(1, num_views - 1))
            else:  # 'mixed' - default for best coverage
                # Mix of elevations: bottom, side, angled, and top views
                # Pattern includes NEGATIVE elevations for bottom views
                # This ensures COMPLETE coverage including the bottom surface
                if num_views >= 16:
                    # For 16+ views: balanced coverage of all angles
                    # Pattern: [-30°, 30°, 30°, 60°, -30°, 30°, 30°, 75°, ...] repeating
                    elevation_pattern = [-30, 30, 30, 60, -30, 30, 30, 75]
                    elev = elevation_pattern[i % len(elevation_pattern)]
                elif num_views >= 8:
                    # For 8-15 views: include some bottom views
                    # Pattern: [-30°, 30°, 60°, 30°, -30°, 30°, 60°, 75°]
                    elevation_pattern = [-30, 30, 60, 30, -30, 30, 60, 75]
                    elev = elevation_pattern[i % len(elevation_pattern)]
                elif num_views >= 4:
                    # For 4-7 views: alternate between bottom and top
                    elev = -30 if i % 4 == 0 else (60 if i % 2 == 1 else 30)
                else:
                    # For fewer views: use default
                    elev = elevation

            camera = create_camera(
                azimuth=azimuth,
                elevation=elev,
                distance=distance,
                image_size=image_size,
                device=device
            )
            cameras.append(camera)
            camera_configs.append({
                'azimuth': azimuth,
                'elevation': elev,
                'distance': distance,
                'fov': camera_config['fov'],
                'image_size': image_size
            })

        print(
            f"  Created {num_views} cameras (elevation mode: {elevation_mode})")

        # Save camera config
        camera_config_path = output_dir / "camera_config.json"
        save_camera_params(cameras, str(camera_config_path))
        print(f"  Saved: {camera_config_path}")

        # 4. Render target images
        print("\n[4/5] Rendering target images...")

        # Get target types to generate from config
        target_types = rendering_config.get('target_types', ['soft'])
        if not isinstance(target_types, list):
            target_types = [target_types]

        print(f"  Target types: {', '.join(target_types)}")

        gt_vertices = gt_vertices.to(device)
        gt_faces = gt_faces.to(device)

        # Generate targets for each type
        for target_type in target_types:
            is_soft = (target_type == 'soft')
            target_subdir = targets_dir.parent / f"targets_{target_type}"
            target_subdir.mkdir(parents=True, exist_ok=True)

            # Use appropriate renderer based on target type
            if is_soft:
                # Soft targets: Use DifferentiableRenderer with configurable blur
                blur_radius = rendering_config.get('blur_radius_soft', 0.01)
                renderer = DifferentiableRenderer(
                    image_size=image_size,
                    blur_radius=blur_radius,
                    device=device
                )
                print(
                    f"  Generating {target_type} targets (blur_radius={blur_radius})...")
            else:
                # Hard targets: Use ClassicRenderer for clean binary silhouettes
                renderer = ClassicRenderer(
                    image_size=image_size,
                    device=device
                )
                print(f"  Generating {target_type} targets (binary)...")

            for i, camera in enumerate(cameras):
                silhouette = renderer.render_silhouette(
                    gt_vertices,
                    gt_faces,
                    camera
                )

                img_path = target_subdir / f"view_{i:02d}.png"
                save_rendered_image(silhouette, str(img_path))

            print(f"    Saved to: {target_subdir}")

        print(f"  Rendered {num_views} views for each target type")

        # 5. Save metadata
        print("\n[5/5] Saving metadata...")
        metadata = {
            'test_case': test_case_id,
            'name': test_case_data['description'],
            'difficulty': test_case_data['difficulty'],
            'ground_truth_mesh': str(gt_mesh_path),
            'initial_mesh': str(initial_mesh_path),
            'camera_config': str(camera_config_path),
            'num_views': num_views,
            'image_size': image_size,
            'subdivision_level': subdivision_level,
            'target_types_available': target_types,  # NEW: Track which target types exist
            'deformation': deformation
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved: {metadata_path}")

        # Create visualization
        print("\n[6/6] Creating visualization...")
        visualize_test_case(str(output_dir))

        print(f"\n✓ {test_case_id} generation complete!")

    print("\n" + "="*70)
    print("ALL TEST CASES GENERATED SUCCESSFULLY!")
    print("="*70)
    print(f"\nGenerated test cases:")
    for test_case_id, test_case_data in test_cases_config.items():
        print(
            f"  • {test_case_id}: {test_case_data['description']} [{test_case_data['difficulty'].upper()}]")
    print(f"\nConfiguration used:")
    print(f"  • Views per test case: {rendering_config['num_views']}")
    print(
        f"  • Image size: {rendering_config['image_size']}x{rendering_config['image_size']}")
    print(f"  • Subdivision level: {mesh_config['subdivision_level']}")
    print()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("3D RECONSTRUCTION - TARGET GENERATION")
    print("="*70)
    print("\nChoose generation mode:")
    print("  1. Use generation_config.yaml (recommended - configurable)")
    print("  2. Use legacy functions (original hard-coded settings)")
    print()

    mode = input("Enter mode (1 or 2, default=1): ").strip()

    if mode == "2":
        # Legacy mode - use original functions
        print("\nUsing legacy generation functions...")
        print("This script generates synthetic test cases for 3D reconstruction.")
        print("Each test case includes:")
        print("  • Ground truth 3D mesh")
        print("  • Initial sphere mesh")
        print("  • Multiple camera viewpoints")
        print("  • Target rendered images (silhouettes)")
        print("  • Configuration metadata")
        print()

        # Generate all test cases
        try:
            generate_test_case_a()
            visualize_test_case("data/test_cases/test_case_a")

            generate_test_case_b()
            visualize_test_case("data/test_cases/test_case_b")

            generate_test_case_c()
            visualize_test_case("data/test_cases/test_case_c")

        except Exception as e:
            print(f"\n✗ Error during generation: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        print("\nALL TEST CASES GENERATED SUCCESSFULLY!")
        print("="*70)
        print("\nSummary:")
        print("  • Test Case A: Sphere → Ellipsoid (4 views) [EASY]")
        print("  • Test Case B: Sphere → Bump (8 views) [MEDIUM]")
        print("  • Test Case C: Complex Shape (12 views) [HARD]")
    else:
        # Config-based mode (default)
        print("\nUsing generation_config.yaml...")
        try:
            config_path = Path("data/generation_config.yaml")
            if not config_path.exists():
                print(f"✗ Config file not found: {config_path}")
                print(
                    "Please create generation_config.yaml or use mode 2 for legacy generation.")
                sys.exit(1)

            generate_from_config(str(config_path))

        except Exception as e:
            print(f"\n✗ Error during generation: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print("\nOutput structure:")
    print("  test_case_[a/b/c]/")
    print("    ├── ground_truth.obj       (target mesh)")
    print("    ├── initial.obj            (starting sphere)")
    print("    ├── camera_config.json     (camera parameters)")
    print("    ├── metadata.json          (test case info)")
    print("    ├── visualization.png      (all views)")
    print("    └── targets/")
    print("        ├── view_00.png")
    print("        ├── view_01.png")
    print("        └── ...")
    print("\nNext steps:")
    print("  1. Examine the generated images in test_case_*/targets/")
    print("  2. Open the .obj files in MeshLab or Blender")
    print("  3. Study the visualization.png in each test case folder")
    print("  4. Run the optimization pipeline!")
    print()
