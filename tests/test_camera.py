"""
Unit tests for camera module.

Tests camera creation, transformations, and multi-view setup.
"""

from src.camera import (
    camera_position_from_spherical,
    create_camera,
    create_camera_circle,
    save_camera_params,
    load_camera_params
)
import torch
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCameraTransforms:
    """Test camera coordinate transformations."""

    def test_spherical_to_cartesian(self):
        """Test spherical to Cartesian conversion."""
        # Test basic cases
        x, y, z = camera_position_from_spherical(0, 0, 1.0)
        assert abs(x - 0.0) < 1e-5
        assert abs(y - 0.0) < 1e-5
        assert abs(z - 1.0) < 1e-5

        # Test with elevation
        x, y, z = camera_position_from_spherical(0, 90, 1.0)
        assert abs(y - 1.0) < 1e-5

    def test_distance_preserved(self):
        """Test that distance from origin is preserved."""
        distance = 2.5
        x, y, z = camera_position_from_spherical(45, 30, distance)
        computed_distance = (x**2 + y**2 + z**2)**0.5
        assert abs(computed_distance - distance) < 1e-5


class TestCameraCreation:
    """Test camera object creation."""

    def test_create_single_camera(self):
        """Test creating a single camera."""
        camera = create_camera(azimuth=0, elevation=30, distance=3.0)
        assert camera is not None

    def test_create_camera_circle(self):
        """Test creating multiple cameras in a circle."""
        cameras = create_camera_circle(num_views=4, elevation=30, distance=3.0)
        assert len(cameras) == 4

        # Test with different number of views
        cameras = create_camera_circle(num_views=8)
        assert len(cameras) == 8


class TestCameraIO:
    """Test camera save/load functionality."""

    def test_save_load_camera_params(self, tmp_path):
        """Test saving and loading camera parameters."""
        # Create cameras
        cameras = create_camera_circle(num_views=4)

        # Save
        save_path = tmp_path / "test_cameras.json"
        save_camera_params(
            cameras,
            path=str(save_path),
            metadata={'test': True}
        )

        assert save_path.exists()

        # Load
        loaded_cameras = load_camera_params(str(save_path))
        assert len(loaded_cameras) == len(cameras)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
