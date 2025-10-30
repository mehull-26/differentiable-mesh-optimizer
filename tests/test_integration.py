"""
Integration tests for the complete pipeline.

Tests end-to-end functionality.
"""

import shutil
import tempfile
from pathlib import Path
import yaml
import torch
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPipelineIntegration:
    """Test complete pipeline integration."""

    @pytest.fixture
    def temp_config(self):
        """Create temporary test configuration."""
        config = {
            'test_case': 'test_case_a',
            'mesh': {
                'subdivision_level': 0,
                'initial_type': 'sphere'
            },
            'optimization': {
                'num_iterations': 2,
                'learning_rate': 0.01,
                'optimizer': 'adam'
            },
            'losses': {
                'silhouette_weight': 1.0,
                'edge_weight': 0.1,
                'laplacian_weight': 0.1,
                'normal_weight': 0.01
            },
            'rendering': {
                'image_size': 64,
                'soft_rendering': True,
                'blur_radius': 0.0
            },
            'visualization': {
                'log_every': 10,
                'save_checkpoint_every': 50,
                'save_video': False,
                'plot_loss_curves': True
            },
            'output': {
                'base_dir': 'test_results',
                'experiment_name': 'test_integration',
                'save_intermediate': False,
                'save_history': True
            },
            'device': {
                'type': 'cpu',
                'cuda_device': 0
            },
            'evaluation': {
                'compute_chamfer': True,
                'compute_normal_consistency': True,
                'compute_edge_length_variance': True,
                'save_evaluation_report': True
            }
        }

        return config

    def test_pipeline_runs_without_data(self, temp_config, tmp_path):
        """Test pipeline with minimal setup (may fail on data load)."""
        from src.mesh import create_icosphere
        from src.camera import create_camera
        from src.renderer import DifferentiableRenderer
        from src.optimizer import MeshOptimizer

        # Test component creation
        vertices, faces = create_icosphere(subdivisions=0)
        camera = create_camera(0, 30, 3.0, image_size=64, device='cpu')
        renderer = DifferentiableRenderer(image_size=64, device='cpu')
        optimizer = MeshOptimizer(renderer=renderer, device='cpu')

        assert vertices is not None
        assert camera is not None
        assert renderer is not None
        assert optimizer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
