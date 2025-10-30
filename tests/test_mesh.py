"""
Unit tests for mesh.py

Run with: pytest tests/test_mesh.py
Or: python -m pytest tests/
"""

from src.mesh import (
    create_icosphere,
    create_deformed_sphere,
    save_mesh,
    load_mesh
)
import tempfile
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_create_icosphere():
    """Test icosphere creation"""
    vertices, faces = create_icosphere(subdivisions=0)

    # Should have 12 vertices for base icosahedron
    assert vertices.shape[0] == 12, f"Expected 12 vertices, got {vertices.shape[0]}"
    assert faces.shape[0] == 20, f"Expected 20 faces, got {faces.shape[0]}"

    # Vertices should be on unit sphere
    distances = torch.norm(vertices, dim=1)
    assert torch.allclose(distances, torch.ones_like(distances), atol=1e-5), \
        "Vertices should be on unit sphere"


def test_deformed_sphere_ellipsoid():
    """Test ellipsoid deformation"""
    vertices, faces = create_deformed_sphere(
        'ellipsoid', {'scale': [2.0, 1.0, 1.0]})

    # Check x-coordinates are stretched
    max_x = vertices[:, 0].max().item()
    max_y = vertices[:, 1].max().item()
    assert max_x > max_y * \
        1.5, f"X ({max_x}) should be stretched more than Y ({max_y})"


def test_deformed_sphere_bump():
    """Test bump deformation"""
    vertices, faces = create_deformed_sphere('bump', {'amplitude': 0.5})

    # Vertices with x > 0 should be pushed outward
    original_verts, _ = create_icosphere(subdivisions=2)

    positive_x_mask = original_verts[:, 0] > 0
    if positive_x_mask.any():
        # Check that some vertices moved in +x direction
        diff = vertices[positive_x_mask, 0] - \
            original_verts[positive_x_mask, 0]
        assert (diff > 0).any(), "Bump should push some vertices outward"


def test_save_load_mesh():
    """Test save/load round trip"""
    vertices, faces = create_icosphere()

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
        temp_path = f.name

    try:
        save_mesh(vertices, faces, temp_path)
        loaded_v, loaded_f = load_mesh(temp_path)

        # Check if loaded mesh matches original
        assert torch.allclose(vertices, loaded_v,
                              atol=1e-5), "Vertices don't match"
        assert torch.equal(faces, loaded_f), "Faces don't match"
    finally:
        os.unlink(temp_path)


def test_mesh_has_correct_types():
    """Test that tensors have correct dtypes"""
    vertices, faces = create_icosphere()

    assert vertices.dtype == torch.float32, "Vertices should be float32"
    assert faces.dtype == torch.int64, "Faces should be int64"


if __name__ == "__main__":
    # Can still run directly for quick checks
    import pytest
    pytest.main([__file__, "-v"])
