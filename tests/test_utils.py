"""
Unit tests for utility functions.

Tests visualization and analysis utilities.
"""

from src.utils import (
    compute_vertex_normals,
    compute_edge_length_variance,
    compare_meshes,
    setup_experiment_directory
)
from src.mesh import create_icosphere
from pathlib import Path
import torch
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestVertexNormals:
    """Test vertex normal computation."""

    def test_compute_normals_sphere(self):
        """Test normal computation on sphere."""
        vertices, faces = create_icosphere(subdivisions=1)
        normals = compute_vertex_normals(vertices, faces)

        assert normals.shape == vertices.shape

        # Check normalization
        norms = normals.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestEdgeVariance:
    """Test edge length variance computation."""

    def test_edge_variance_sphere(self):
        """Test edge variance on sphere."""
        vertices, faces = create_icosphere(subdivisions=1)
        variance = compute_edge_length_variance(vertices, faces)

        assert variance >= 0
        assert isinstance(variance, float)


class TestMeshComparison:
    """Test mesh comparison metrics."""

    def test_compare_identical_meshes(self):
        """Test comparing identical meshes."""
        vertices, faces = create_icosphere(subdivisions=1)

        metrics = compare_meshes(
            vertices, vertices, faces,
            compute_chamfer=True,
            compute_normal_consistency=True,
            compute_edge_variance=True
        )

        assert 'chamfer_distance' in metrics
        assert metrics['chamfer_distance'] < 1e-5  # Should be ~0
        assert metrics['normal_consistency'] > 0.99  # Should be ~1


class TestExperimentSetup:
    """Test experiment directory setup."""

    def test_setup_experiment_directory(self, tmp_path):
        """Test creating experiment directory."""
        config = {
            'output': {
                'base_dir': str(tmp_path),
                'experiment_name': 'test_exp'
            },
            'test_case': 'test_case_a'
        }

        exp_dir = setup_experiment_directory(config)

        assert exp_dir.exists()
        assert exp_dir.name == 'test_exp'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
