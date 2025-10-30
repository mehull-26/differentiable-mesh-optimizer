"""
Unit tests for loss functions.

Tests all loss computation functions.
"""

from src.losses import (
    silhouette_loss,
    rgb_loss,
    edge_loss,
    laplacian_loss,
    normal_consistency_loss,
    combined_loss
)
from src.mesh import create_icosphere
import torch
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSilhouetteLoss:
    """Test silhouette loss computation."""

    def test_silhouette_loss_perfect_match(self):
        """Test loss is zero for perfect match."""
        rendered = torch.ones(1, 64, 64, 1)
        target = torch.ones(1, 64, 64, 1)

        loss = silhouette_loss(rendered, target, loss_type='l2')
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_silhouette_loss_types(self):
        """Test different loss types."""
        rendered = torch.rand(1, 64, 64, 1)
        target = torch.rand(1, 64, 64, 1)

        loss_l2 = silhouette_loss(rendered, target, loss_type='l2')
        loss_bce = silhouette_loss(rendered, target, loss_type='bce')

        assert loss_l2 > 0
        assert loss_bce > 0


class TestRGBLoss:
    """Test RGB loss computation."""

    def test_rgb_loss_with_mask(self):
        """Test RGB loss with mask."""
        rendered = torch.rand(1, 64, 64, 3)
        target = torch.rand(1, 64, 64, 3)
        mask = torch.ones(1, 64, 64, 1)

        loss = rgb_loss(rendered, target, mask)
        assert loss > 0


class TestEdgeLoss:
    """Test edge length loss."""

    def test_edge_loss_sphere(self):
        """Test edge loss on sphere."""
        vertices, faces = create_icosphere(subdivisions=1)
        loss = edge_loss(vertices, faces)

        assert loss >= 0

    def test_edge_loss_increases_with_stretch(self):
        """Test that stretching increases edge loss."""
        vertices, faces = create_icosphere(subdivisions=1)

        # Original loss
        loss_original = edge_loss(vertices, faces)

        # Stretch mesh
        vertices_stretched = vertices.clone()
        vertices_stretched[:, 0] *= 2.0  # Stretch in x
        loss_stretched = edge_loss(vertices_stretched, faces)

        assert loss_stretched > loss_original


class TestLaplacianLoss:
    """Test Laplacian smoothness loss."""

    def test_laplacian_loss_sphere(self):
        """Test Laplacian loss on sphere."""
        vertices, faces = create_icosphere(subdivisions=1)
        loss = laplacian_loss(vertices, faces)

        assert loss >= 0

    def test_laplacian_loss_increases_with_noise(self):
        """Test that noise increases Laplacian loss."""
        vertices, faces = create_icosphere(subdivisions=1)

        # Original loss
        loss_original = laplacian_loss(vertices, faces)

        # Add noise
        vertices_noisy = vertices + torch.randn_like(vertices) * 0.1
        loss_noisy = laplacian_loss(vertices_noisy, faces)

        assert loss_noisy > loss_original


class TestNormalConsistencyLoss:
    """Test normal consistency loss."""

    def test_normal_consistency_sphere(self):
        """Test normal consistency on sphere."""
        vertices, faces = create_icosphere(subdivisions=1)
        loss = normal_consistency_loss(vertices, faces)

        assert loss >= 0


class TestCombinedLoss:
    """Test combined loss function."""

    def test_combined_loss(self):
        """Test combined loss computation."""
        vertices, faces = create_icosphere(subdivisions=1)
        rendered = torch.rand(1, 64, 64, 1)
        target = torch.rand(1, 64, 64, 1)

        weights = {
            'silhouette': 1.0,
            'edge': 0.1,
            'laplacian': 0.1,
            'normal': 0.01
        }

        loss, loss_dict = combined_loss(
            rendered, target, vertices, faces, weights)
        assert loss > 0
        assert 'total' in loss_dict


class TestGradients:
    """Test gradient flow through losses."""

    def test_edge_loss_gradients(self):
        """Test gradients flow through edge loss."""
        vertices, faces = create_icosphere(subdivisions=1)
        vertices = vertices.clone().requires_grad_(True)

        loss = edge_loss(vertices, faces)
        loss.backward()

        assert vertices.grad is not None
        assert vertices.grad.abs().sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
