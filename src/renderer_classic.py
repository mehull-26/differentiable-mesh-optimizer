"""
Classic (Non-Differentiable) Renderer for Target Generation

This renderer uses PyTorch3D's hard rasterization without soft blending.
Perfect for generating clean, binary silhouette targets without gradient artifacts.

Key differences from DifferentiableRenderer:
- No blur/softness - pure binary silhouettes
- No gradient support needed (faster)
- Used only for target image generation
- Clean, crisp edges without anti-aliasing artifacts

When to use:
- Generating hard target images (targets_hard/)
- When you need clean binary masks
- NOT for optimization (use DifferentiableRenderer instead)
"""

import torch
import numpy as np
from typing import Tuple, Optional
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
    HardPhongShader,
    PointLights,
    MeshRenderer,
)


class ClassicRenderer:
    """
    Classic non-differentiable renderer for clean binary silhouettes.

    This renderer produces hard-edged, binary silhouettes suitable for
    target image generation. No soft edges, no gradients, just clean masks.
    """

    def __init__(self,
                 image_size: int = 256,
                 device: str = 'cpu'):
        """
        Initialize the classic renderer.

        Args:
            image_size: Output image resolution (square images)
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.image_size = image_size

        # Hard rasterization settings - no blur, binary output
        self.raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,  # No blur = hard binary edges
            faces_per_pixel=1,  # Only closest face
            perspective_correct=True,
            bin_size=0,  # Auto bin size
        )

        print(f"Initialized ClassicRenderer:")
        print(f"  - Image size: {image_size}x{image_size}")
        print(f"  - Device: {device}")
        print(f"  - Mode: Binary (no blur)")

    def render_silhouette(self,
                          vertices: torch.Tensor,
                          faces: torch.Tensor,
                          cameras) -> torch.Tensor:
        """
        Render clean binary silhouette (hard edges, no gradients).

        Args:
            vertices: (V, 3) vertex positions
            faces: (F, 3) triangle indices
            cameras: PyTorch3D camera object

        Returns:
            silhouette: (1, H, W, 1) tensor with binary values (0 or 1)
        """
        # Create PyTorch3D mesh structure
        mesh = Meshes(
            verts=[vertices],
            faces=[faces]
        ).to(self.device)

        # Create rasterizer
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=self.raster_settings
        )

        # Rasterize to get fragments
        with torch.no_grad():  # No gradients needed for target generation
            fragments = rasterizer(mesh)

        # Extract binary silhouette from fragments
        # pix_to_face contains face indices, -1 for background
        pix_to_face = fragments.pix_to_face[..., 0]  # (1, H, W)

        # Create binary mask: 1 where mesh is visible, 0 for background
        silhouette = (pix_to_face >= 0).float().unsqueeze(-1)  # (1, H, W, 1)

        return silhouette

    def render_depth(self,
                     vertices: torch.Tensor,
                     faces: torch.Tensor,
                     cameras) -> torch.Tensor:
        """
        Render depth map (distance from camera to surface).

        Args:
            vertices: (V, 3) vertex positions
            faces: (F, 3) triangle indices
            cameras: PyTorch3D camera object

        Returns:
            depth: (1, H, W, 1) tensor with depth values
        """
        mesh = Meshes(
            verts=[vertices],
            faces=[faces]
        ).to(self.device)

        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=self.raster_settings
        )

        with torch.no_grad():
            fragments = rasterizer(mesh)

        # Extract depth
        depth = fragments.zbuf[..., 0:1]  # (1, H, W, 1)

        # Handle background (set to large value)
        depth = torch.where(
            depth < 0,
            torch.ones_like(depth) * 100.0,
            depth
        )

        return depth


def save_rendered_image(image: torch.Tensor, path: str):
    """
    Save rendered image to file.

    Args:
        image: (1, H, W, C) tensor with values in [0, 1]
        path: Output file path (e.g., 'output.png')
    """
    import matplotlib.pyplot as plt

    # Convert to numpy and remove batch dimension
    img_np = image[0].detach().cpu().numpy()

    # Handle different channel counts
    if img_np.shape[-1] == 1:
        # Grayscale (silhouette or depth)
        img_np = img_np.squeeze(-1)
        plt.imsave(path, img_np, cmap='gray', vmin=0.0, vmax=1.0)
    else:
        # RGB
        plt.imsave(path, np.clip(img_np, 0, 1))


# Quick test
if __name__ == "__main__":
    print("Testing ClassicRenderer...")

    import sys
    import os
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))

    from src.mesh import create_icosphere
    from src.camera import create_camera

    # Create mesh
    vertices, faces = create_icosphere(subdivisions=1)

    # Create camera
    camera = create_camera(azimuth=45, elevation=30,
                           distance=3.0, device='cpu')

    # Create renderer
    renderer = ClassicRenderer(image_size=256, device='cpu')

    # Render
    silhouette = renderer.render_silhouette(vertices, faces, camera)

    print(f"\nRendered silhouette:")
    print(f"  Shape: {silhouette.shape}")
    print(f"  Min: {silhouette.min():.3f}, Max: {silhouette.max():.3f}")
    print(f"  Unique values: {torch.unique(silhouette).tolist()}")
    print(
        f"  Binary pixels (0 or 1): {((silhouette == 0) | (silhouette == 1)).sum().item()} / {silhouette.numel()}")

    # Save test image
    os.makedirs("tests/renders", exist_ok=True)
    save_rendered_image(silhouette, "tests/renders/classic_test.png")
    print(f"\n✓ Saved test render to tests/renders/classic_test.png")
    print("✓ ClassicRenderer test passed!")
