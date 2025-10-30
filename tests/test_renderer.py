"""
Unit tests for renderer module.

Tests differentiable rendering functionality.
"""

from src.renderer import DifferentiableRenderer
from src.camera import create_camera
from src.mesh import create_icosphere
import torch
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRendererInitialization:
    """Test renderer initialization."""

    def test_create_renderer_cpu(self):
        """Test creating renderer on CPU."""
        renderer = DifferentiableRenderer(image_size=64, device='cpu')
        assert renderer is not None
        assert renderer.device == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_create_renderer_cuda(self):
        """Test creating renderer on CUDA."""
        renderer = DifferentiableRenderer(image_size=64, device='cuda')
        assert renderer is not None
        assert 'cuda' in renderer.device


class TestSilhouetteRendering:
    """Test silhouette rendering."""

    def test_render_silhouette_shape(self):
        """Test output shape of silhouette rendering."""
        renderer = DifferentiableRenderer(image_size=64, device='cpu')
        vertices, faces = create_icosphere(subdivisions=1)
        camera = create_camera(0, 30, 3.0, image_size=64, device='cpu')

        silhouette = renderer.render_silhouette(vertices, faces, camera)

        assert silhouette.shape == (1, 64, 64, 1)
        assert silhouette.min() >= 0.0
        assert silhouette.max() <= 1.0

    def test_render_soft_vs_hard(self):
        """Test that soft and hard rendering produce different results."""
        from src.renderer import ClassicRenderer

        soft_renderer = DifferentiableRenderer(
            image_size=64, device='cpu', blur_radius=0.01)
        hard_renderer = ClassicRenderer(image_size=64, device='cpu')

        vertices, faces = create_icosphere(subdivisions=1)
        camera = create_camera(0, 30, 3.0, image_size=64, device='cpu')

        soft = soft_renderer.render_silhouette(vertices, faces, camera)
        hard = hard_renderer.render_silhouette(vertices, faces, camera)

        # Should be different at edges (soft has blur, hard is binary)
        assert not torch.allclose(soft, hard)


class TestDepthRendering:
    """Test depth rendering."""

    def test_render_depth_shape(self):
        """Test output shape of depth rendering."""
        renderer = DifferentiableRenderer(image_size=64, device='cpu')
        vertices, faces = create_icosphere(subdivisions=1)
        camera = create_camera(0, 30, 3.0, image_size=64, device='cpu')

        depth = renderer.render_depth(vertices, faces, camera)

        assert depth.shape == (1, 64, 64, 1)


class TestDifferentiability:
    """Test gradient flow through rendering."""

    def test_silhouette_gradients(self):
        """Test that gradients flow through silhouette rendering."""
        renderer = DifferentiableRenderer(
            image_size=64, device='cpu', blur_radius=0.01)
        vertices, faces = create_icosphere(subdivisions=1)
        vertices = vertices.clone().requires_grad_(True)
        camera = create_camera(0, 30, 3.0, image_size=64, device='cpu')

        silhouette = renderer.render_silhouette(vertices, faces, camera)
        loss = silhouette.sum()
        loss.backward()

        assert vertices.grad is not None
        assert vertices.grad.abs().sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
