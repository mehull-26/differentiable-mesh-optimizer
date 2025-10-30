import torch
import numpy as np
import json
from typing import Tuple, Dict, List, Union
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
)


def create_camera(azimuth: float,
                  elevation: float,
                  distance: float,
                  image_size: int = 256,
                  device: str = 'cpu') -> FoVPerspectiveCameras:
    """
    Create a PyTorch3D camera from spherical coordinates.

    Spherical coordinates explanation:
    - Azimuth: Rotation around the vertical (y) axis in degrees
               0° = looking from +Z axis (front)
               90° = looking from +X axis (right side)
               180° = looking from -Z axis (back)
               270° = looking from -X axis (left side)

    - Elevation: Angle above/below the horizontal plane in degrees
                 0° = looking horizontally at the equator
                 90° = looking down from directly above (top view)
                 -90° = looking up from directly below (bottom view)

    - Distance: How far the camera is from the origin (world center)
                Larger distance = camera further away (smaller object in view)

    Args:
        azimuth: Horizontal rotation angle in degrees (0-360)
        elevation: Vertical angle in degrees (-90 to 90)
        distance: Distance from origin to camera
        image_size: Size of rendered image in pixels (square images)
        device: 'cpu' or 'cuda'

    Returns:
        PyTorch3D camera object ready for rendering

    Example:
        # Camera looking from the front, slightly above, 3 units away
        camera = create_camera(azimuth=0, elevation=30, distance=3.0)

        # Camera looking from the right side
        camera = create_camera(azimuth=90, elevation=0, distance=3.0)
    """
    # Convert spherical coordinates to camera position and orientation
    # PyTorch3D's look_at_view_transform handles the conversion
    R, T = look_at_view_transform(
        dist=distance,      # Distance from origin
        elev=elevation,     # Elevation angle
        azim=azimuth,       # Azimuth angle
        device=device
    )

    # Create perspective camera with field of view
    # fov (field of view): Controls how "zoomed in" the camera is
    #   - Smaller fov (e.g., 30°) = more zoomed in (telephoto lens)
    #   - Larger fov (e.g., 90°) = more zoomed out (wide angle lens)
    #   - 60° is a reasonable default for human-like perspective
    cameras = FoVPerspectiveCameras(
        device=device,
        R=R,                    # Rotation matrix (3x3)
        T=T,                    # Translation vector (3,)
        fov=60,                 # Field of view in degrees
    )

    return cameras


def create_camera_circle(num_views: int = 8,
                         elevation: float = 30.0,
                         distance: float = 3.0,
                         image_size: int = 256,
                         device: str = 'cpu') -> List[FoVPerspectiveCameras]:
    """
    Create multiple cameras arranged in a circle around the object.
    Useful for multi-view reconstruction.

    Args:
        num_views: Number of evenly spaced camera positions
        elevation: Fixed elevation angle for all cameras
        distance: Fixed distance from origin for all cameras
        image_size: Image size in pixels
        device: 'cpu' or 'cuda'

    Returns:
        List of camera objects

    Example:
        # Create 8 cameras around the object
        cameras = create_camera_circle(num_views=8, elevation=30, distance=3.0)
        # cameras[0] looks from front (azimuth=0)
        # cameras[2] looks from right (azimuth=90)
        # cameras[4] looks from back (azimuth=180)
    """
    cameras = []

    # Create cameras at evenly spaced azimuth angles
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

    return cameras


def camera_position_from_spherical(azimuth: float,
                                   elevation: float,
                                   distance: float) -> Tuple[float, float, float]:
    """
    Convert spherical coordinates to Cartesian 3D position.
    Useful for understanding where the camera is in 3D space.

    Math reference:
        x = distance * cos(elevation) * sin(azimuth)
        y = distance * sin(elevation)
        z = distance * cos(elevation) * cos(azimuth)

    Args:
        azimuth: Angle in degrees
        elevation: Angle in degrees
        distance: Distance from origin

    Returns:
        (x, y, z) position tuple

    Example:
        >>> pos = camera_position_from_spherical(0, 0, 3.0)
        >>> print(f"Camera at: {pos}")  # (0.0, 0.0, 3.0) - on +Z axis

        >>> pos = camera_position_from_spherical(90, 0, 3.0)
        >>> print(f"Camera at: {pos}")  # (3.0, 0.0, 0.0) - on +X axis
    """
    # Convert degrees to radians
    azim_rad = np.radians(azimuth)
    elev_rad = np.radians(elevation)

    # Spherical to Cartesian conversion
    x = distance * np.cos(elev_rad) * np.sin(azim_rad)
    y = distance * np.sin(elev_rad)
    z = distance * np.cos(elev_rad) * np.cos(azim_rad)

    return (float(x), float(y), float(z))


def save_camera_params(cameras: Union[FoVPerspectiveCameras, List[FoVPerspectiveCameras]],
                       path: str,
                       metadata: Dict = None):
    """
    Save camera parameters to JSON file for reproducibility.
    This is important for:
    - Reproducing experiments
    - Sharing camera configurations
    - Loading camera setups later

    Args:
        cameras: Single camera or list of cameras
        path: Output JSON file path
        metadata: Optional dict with additional info (e.g., image_size, dataset name)

    Example:
        camera = create_camera(azimuth=45, elevation=30, distance=3.0)
        save_camera_params(camera, "config/camera_config.json", 
                          metadata={'image_size': 256, 'notes': 'Test setup'})
    """
    # Handle both single camera and list of cameras
    if not isinstance(cameras, list):
        cameras = [cameras]

    camera_data = []

    for i, cam in enumerate(cameras):
        # Extract rotation matrix and translation vector
        R = cam.R.cpu().numpy()  # (1, 3, 3) or (3, 3)
        T = cam.T.cpu().numpy()  # (1, 3) or (3,)

        # Handle batch dimension
        if R.ndim == 3:
            R = R[0]  # Take first element if batched
        if T.ndim == 2:
            T = T[0]

        # Store camera parameters
        cam_dict = {
            'camera_id': i,
            'rotation_matrix': R.tolist(),      # 3x3 rotation
            'translation': T.tolist(),          # 3D translation
            'fov': float(cam.fov) if hasattr(cam, 'fov') else 60.0,
        }

        camera_data.append(cam_dict)

    # Create output structure
    output = {
        'num_cameras': len(cameras),
        'cameras': camera_data,
        'metadata': metadata or {}
    }

    # Save to JSON file
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(cameras)} camera(s) to {path}")


def load_camera_params(path: str, device: str = 'cpu') -> List[FoVPerspectiveCameras]:
    """
    Load camera configuration from JSON file.

    Args:
        path: Input JSON file path
        device: 'cpu' or 'cuda'

    Returns:
        List of reconstructed camera objects

    Example:
        cameras = load_camera_params("config/camera_config.json")
        # Use cameras for rendering or optimization
    """
    with open(path, 'r') as f:
        data = json.load(f)

    cameras = []

    for cam_data in data['cameras']:
        # Reconstruct rotation and translation as tensors
        R = torch.tensor(cam_data['rotation_matrix'],
                         dtype=torch.float32, device=device)
        T = torch.tensor(cam_data['translation'],
                         dtype=torch.float32, device=device)

        # Add batch dimension if needed
        if R.dim() == 2:
            R = R.unsqueeze(0)  # (3, 3) -> (1, 3, 3)
        if T.dim() == 1:
            T = T.unsqueeze(0)  # (3,) -> (1, 3)

        # Recreate camera
        camera = FoVPerspectiveCameras(
            device=device,
            R=R,
            T=T,
            fov=cam_data.get('fov', 60.0)
        )

        cameras.append(camera)

    print(f"Loaded {len(cameras)} camera(s) from {path}")
    return cameras


def visualize_camera_positions(cameras: List[FoVPerspectiveCameras],
                               save_path: str = None):
    """
    Create a simple visualization of camera positions around the origin.
    Helpful for debugging and understanding camera placement.

    Args:
        cameras: List of camera objects
        save_path: Optional path to save visualization (requires matplotlib)
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib not installed. Skipping visualization.")
        return

    # Extract camera positions from translation vectors
    positions = []
    for cam in cameras:
        T = cam.T.cpu().numpy()
        if T.ndim == 2:
            T = T[0]
        # In PyTorch3D, camera position is at -R^T @ T
        # But for visualization, T is sufficient
        positions.append(-T)  # Negate to get actual camera position

    positions = np.array(positions)

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot object at origin
    ax.scatter([0], [0], [0], c='red', marker='o',
               s=200, label='Object (origin)')

    # Plot camera positions
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c='blue', marker='^', s=100, label='Cameras')

    # Draw lines from origin to cameras
    for i, pos in enumerate(positions):
        ax.plot([0, pos[0]], [0, pos[1]], [0, pos[2]], 'b--', alpha=0.3)
        ax.text(pos[0], pos[1], pos[2], f'  Cam{i}', fontsize=8)

    # Labels and styling
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Positions Around Object')
    ax.legend()

    # Equal aspect ratio
    max_range = np.array([positions[:, 0].max()-positions[:, 0].min(),
                          positions[:, 1].max()-positions[:, 1].min(),
                          positions[:, 2].max()-positions[:, 2].min()]).max() / 2.0
    mid_x = (positions[:, 0].max()+positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max()+positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max()+positions[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved camera visualization to {save_path}")
    else:
        plt.show()

    plt.close()


# ==================== UNIT TESTS ====================

def test_create_camera():
    """Test basic camera creation"""
    camera = create_camera(azimuth=0, elevation=30, distance=3.0)

    # Check that camera has required attributes
    assert hasattr(camera, 'R'), "Camera should have rotation matrix R"
    assert hasattr(camera, 'T'), "Camera should have translation vector T"

    # Check shapes
    assert camera.R.shape[-2:] == (3, 3), "Rotation should be 3x3"
    assert camera.T.shape[-1] == 3, "Translation should be 3D"

    print("✓ test_create_camera passed")


def test_camera_circle():
    """Test multi-view camera creation"""
    cameras = create_camera_circle(num_views=8, elevation=30, distance=3.0)

    assert len(cameras) == 8, f"Expected 8 cameras, got {len(cameras)}"

    # Check that cameras are at different positions
    positions = []
    for cam in cameras:
        T = cam.T.cpu().numpy()
        positions.append(T)

    positions = np.array(positions)
    # Cameras should not all be at the same position
    assert positions.std() > 0.1, "Cameras should be at different positions"

    print("✓ test_camera_circle passed")


def test_spherical_to_cartesian():
    """Test coordinate conversion"""
    # Test known positions
    # Azimuth=0, Elevation=0 should give position on +Z axis
    x, y, z = camera_position_from_spherical(0, 0, 3.0)
    assert abs(x) < 1e-5, f"Expected x≈0, got {x}"
    assert abs(y) < 1e-5, f"Expected y≈0, got {y}"
    assert abs(z - 3.0) < 1e-5, f"Expected z≈3.0, got {z}"

    # Azimuth=90, Elevation=0 should give position on +X axis
    x, y, z = camera_position_from_spherical(90, 0, 3.0)
    assert abs(x - 3.0) < 1e-5, f"Expected x≈3.0, got {x}"
    assert abs(y) < 1e-5, f"Expected y≈0, got {y}"
    assert abs(z) < 1e-5, f"Expected z≈0, got {z}"

    print("✓ test_spherical_to_cartesian passed")


def test_save_load_camera():
    """Test camera save/load round trip"""
    import tempfile
    import os

    # Create test cameras
    cameras = create_camera_circle(num_views=4, elevation=30, distance=3.0)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        save_camera_params(cameras, temp_path, metadata={'test': True})
        loaded_cameras = load_camera_params(temp_path)

        # Check that we got the same number of cameras
        assert len(loaded_cameras) == len(cameras), \
            f"Expected {len(cameras)} cameras, got {len(loaded_cameras)}"

        # Check that rotation matrices are similar
        for i in range(len(cameras)):
            R_orig = cameras[i].R.cpu()
            R_load = loaded_cameras[i].R.cpu()
            assert torch.allclose(R_orig, R_load, atol=1e-5), \
                f"Rotation matrices don't match for camera {i}"

        print("✓ test_save_load_camera passed")
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    # This runs when you execute: python src/camera.py
    print("Testing camera utilities...")
    print()

    # Run tests
    test_create_camera()
    test_camera_circle()
    test_spherical_to_cartesian()
    test_save_load_camera()

    print()
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print()

    # Demo: Visualize camera positions
    print("Creating demo camera configuration...")
    cameras = create_camera_circle(num_views=8, elevation=30, distance=3.0)

    # Print camera positions for educational purposes
    print("\nCamera positions in 3D space:")
    print("-" * 60)
    for i, cam in enumerate(cameras):
        azimuth = i * 45  # 360/8 = 45 degrees apart
        x, y, z = camera_position_from_spherical(azimuth, 30, 3.0)
        print(
            f"Camera {i}: azimuth={azimuth:3.0f}°, position=({x:6.2f}, {y:6.2f}, {z:6.2f})")

    # Save example configuration
    import os
    os.makedirs("config", exist_ok=True)
    save_camera_params(
        cameras,
        "config/example_cameras.json",
        metadata={
            'num_views': 8,
            'elevation': 30,
            'distance': 3.0,
            'image_size': 256,
            'description': 'Example camera configuration for testing'
        }
    )

    print("\n" + "=" * 60)
    print("Demo complete! Check config/example_cameras.json")
    print("=" * 60)
