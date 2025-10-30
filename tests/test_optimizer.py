"""
Unit tests for optimizer module.

Tests mesh optimization functionality.
"""

from src.optimizer import MeshOptimizer
from src.renderer import DifferentiableRenderer
from src.camera import create_camera
from src.mesh import create_icosphere
import torch
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestOptimizerInitialization:
    """Test optimizer initialization."""

    def test_create_optimizer_adam(self):
        """Test creating optimizer with Adam."""
        renderer = DifferentiableRenderer(image_size=64, device='cpu')
        optimizer = MeshOptimizer(
            renderer=renderer,
            learning_rate=0.01,
            optimizer_type='adam',
            device='cpu'
        )

        assert optimizer is not None
        assert optimizer.optimizer_type == 'adam'

    def test_create_optimizer_sgd(self):
        """Test creating optimizer with SGD."""
        renderer = DifferentiableRenderer(image_size=64, device='cpu')
        optimizer = MeshOptimizer(
            renderer=renderer,
            learning_rate=0.01,
            optimizer_type='sgd',
            device='cpu'
        )

        assert optimizer is not None
        assert optimizer.optimizer_type == 'sgd'


class TestOptimizationSetup:
    """Test optimization setup."""

    def test_setup_optimization(self):
        """Test setting up optimization."""
        renderer = DifferentiableRenderer(image_size=64, device='cpu')
        optimizer = MeshOptimizer(renderer=renderer, device='cpu')

        vertices, faces = create_icosphere(subdivisions=1)
        opt_vertices = optimizer.setup_optimization(vertices, faces)

        assert opt_vertices.requires_grad
        assert optimizer.optimizer is not None


class TestOptimizationStep:
    """Test single optimization step."""

    def test_optimization_step_runs(self):
        """Test that optimization step executes without error."""
        renderer = DifferentiableRenderer(image_size=64, device='cpu')
        optimizer = MeshOptimizer(renderer=renderer, device='cpu')

        vertices, faces = create_icosphere(subdivisions=1)
        optimizer.setup_optimization(vertices, faces)

        camera = create_camera(0, 30, 3.0, image_size=64, device='cpu')
        target = torch.ones(1, 64, 64, 1)

        loss_dict = optimizer.optimization_step([target], [camera])

        assert 'total' in loss_dict
        assert 'silhouette' in loss_dict
        assert loss_dict['total'] > 0


class TestOptimizationLoop:
    """Test full optimization loop."""

    def test_optimization_reduces_loss(self):
        """Test that optimization reduces loss over iterations."""
        renderer = DifferentiableRenderer(image_size=64, device='cpu')
        optimizer = MeshOptimizer(
            renderer=renderer,
            learning_rate=0.01,
            device='cpu'
        )

        vertices, faces = create_icosphere(subdivisions=1)
        camera = create_camera(0, 30, 3.0, image_size=64, device='cpu')
        target = torch.ones(1, 64, 64, 1)

        optimized_vertices, history = optimizer.optimize_mesh(
            initial_vertices=vertices,
            faces=faces,
            target_images=[target],
            cameras=[camera],
            num_iterations=5,
            log_interval=10
        )

        # Check that loss decreased
        assert len(history['total_loss']) == 5
        assert history['total_loss'][-1] < history['total_loss'][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
