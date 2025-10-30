"""
Renderer Module - Unified Interface

This module provides both rendering options:
1. DifferentiableRenderer - For optimization (soft, gradients)
2. ClassicRenderer - For target generation (hard, binary)

Import whichever you need, or import both.
"""

# Import both renderers
from src.renderer_differentiable import DifferentiableRenderer, save_rendered_image, visualize_renders
from src.renderer_classic import ClassicRenderer

# Export all
__all__ = [
    'DifferentiableRenderer',  # For optimization with gradients
    'ClassicRenderer',         # For hard target generation
    'save_rendered_image',     # Utility function
    'visualize_renders',       # Utility function
]
