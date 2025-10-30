import torch
import numpy as np
from typing import Tuple, Optional
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    # Rasterization
    RasterizationSettings,
    MeshRasterizer,
    # Shaders
    SoftSilhouetteShader,
    HardPhongShader,
    SoftPhongShader,
    # Blending
    BlendParams,
    # Lighting
    PointLights,
    DirectionalLights,
    AmbientLights,
    # Materials
    Materials,
    # Complete Renderer
    MeshRenderer,
)


class DifferentiableRenderer:
    """
    Differentiable mesh renderer using PyTorch3D.

    What makes it "differentiable"?
    - Gradients can flow from pixel values back to mesh vertices
    - This enables gradient-based optimization (like gradient descent)
    - We can optimize mesh shape to match target images!

    Key concept: In traditional rendering, you can't ask "how should I change
    the 3D mesh to make the rendered image look different?". With differentiable
    rendering, PyTorch's autograd can answer this question!

    Example workflow:
        1. Render mesh → get image
        2. Compare with target image → get loss
        3. Backpropagate through renderer → get gradients on vertices
        4. Update vertices → mesh changes shape
        5. Repeat until mesh matches target
    """

    def __init__(self,
                 image_size: int = 256,
                 faces_per_pixel: int = 1,
                 blur_radius: float = 1e-2,  # Default blur for soft rendering
                 device: str = 'cpu'):
        """
        Initialize the differentiable renderer.

        Args:
            image_size: Output image resolution (square images)
            faces_per_pixel: How many faces to consider per pixel
                            (1 = faster, >1 = smoother gradients)
            blur_radius: Softness of silhouette edges for soft rendering
                        (e.g., 1e-2 = good gradients, 1e-3 = sharper but still soft)
                        This controls the trade-off between gradient quality and edge sharpness
            device: 'cpu' or 'cuda'

        Understanding rasterization settings:
        - image_size: Controls output resolution
        - faces_per_pixel: More faces = better anti-aliasing but slower
        - blur_radius: Critical for differentiability!
                      Larger values = smoother gradients but blurrier edges
                      Smaller values = sharper edges but noisier gradients
        """
        self.device = device
        self.image_size = image_size
        self.blur_radius = blur_radius

        # Rasterization settings for soft rendering (blurred edges, good gradients)
        # Used for optimization and checkpoint visualization
        self.raster_settings_soft = RasterizationSettings(
            image_size=image_size,
            blur_radius=blur_radius,  # Configurable from config
            faces_per_pixel=50,  # Many faces for smooth coverage
            perspective_correct=True,
        )

        print(f"Initialized DifferentiableRenderer:")
        print(f"  - Image size: {image_size}x{image_size}")
        print(f"  - Device: {device}")
        print(f"  - Soft blur radius: {blur_radius}")

    def render_silhouette(self,
                          vertices: torch.Tensor,
                          faces: torch.Tensor,
                          cameras) -> torch.Tensor:
        """
        Render soft silhouette with differentiable gradients.

        This renderer ONLY does soft rendering with blur for smooth gradients.
        For hard binary silhouettes, use ClassicRenderer instead.

        What is a silhouette?
        - Image where 1 (white) = object visible, 0 (black) = background
        - Only cares about shape, not colors or lighting
        - Perfect for optimizing mesh shape from masks/contours

        Why soft rendering:
        - Smooth gradients at edges enable optimization
        - Blur radius controls gradient quality vs edge sharpness trade-off
        - Essential for gradient-based mesh optimization

        Args:
            vertices: (V, 3) vertex positions
            faces: (F, 3) triangle indices
            cameras: PyTorch3D camera object

        Returns:
            silhouette: (1, H, W, 1) tensor with values in [0, 1]
                       1.0 = object, 0.0 = background
                       Edges are smoothly blended based on blur_radius

        Example:
            renderer = DifferentiableRenderer(blur_radius=1e-2)
            silhouette = renderer.render_silhouette(vertices, faces, camera)
            # silhouette is differentiable w.r.t. vertices!
        """
        # Create PyTorch3D mesh structure
        mesh = Meshes(
            verts=[vertices],
            faces=[faces]
        ).to(self.device)

        # Create rasterizer with soft settings
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=self.raster_settings_soft
        )

        # Create soft silhouette shader
        shader = SoftSilhouetteShader()

        # Combine rasterizer + shader = complete renderer
        renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=shader
        )

        # Render with gradients
        images = renderer(mesh)

        # Extract alpha channel (silhouette)
        silhouette = images[..., 3:4]  # (1, H, W, 1)

        return silhouette

    def render_rgb(self,
                   vertices: torch.Tensor,
                   faces: torch.Tensor,
                   cameras,
                   lights=None,
                   vertex_colors: Optional[torch.Tensor] = None,
                   soft: bool = False) -> torch.Tensor:
        """
        Render RGB image with lighting and shading.

        This is more complex than silhouette rendering because it considers:
        - Lighting (where are the lights? how bright?)
        - Shading (how does light interact with surface?)
        - Colors (what color is each vertex/face?)

        When to use RGB rendering vs silhouette:
        - RGB: When you have color images and want to match appearance
        - Silhouette: When you only care about shape (faster, simpler)

        Args:
            vertices: (V, 3) vertex positions
            faces: (F, 3) triangle indices
            cameras: PyTorch3D camera object
            lights: PyTorch3D lights (if None, creates default lighting)
            vertex_colors: (V, 3) RGB colors per vertex (if None, uses white)
            soft: If True, use soft shading (smoother gradients)

        Returns:
            rgb_image: (1, H, W, 3) tensor with RGB values in [0, 1]

        Example:
            renderer = DifferentiableRenderer()
            rgb = renderer.render_rgb(vertices, faces, camera, lights)
        """
        # Default lighting if not provided
        if lights is None:
            # Create a point light at the camera location
            lights = PointLights(
                device=self.device,
                location=[[0.0, 0.0, 3.0]],  # Light position
                ambient_color=((0.5, 0.5, 0.5),),  # Ambient light (everywhere)
                diffuse_color=((0.7, 0.7, 0.7),),  # Diffuse light (surface)
                # Specular highlights (shiny)
                specular_color=((0.2, 0.2, 0.2),)
            )

        # Default vertex colors if not provided (white)
        if vertex_colors is None:
            vertex_colors = torch.ones_like(vertices)  # (V, 3) all white

        # Create mesh with vertex colors
        # In PyTorch3D, we can attach colors as "textures"
        mesh = Meshes(
            verts=[vertices],
            faces=[faces],
        ).to(self.device)

        # We'll use a simpler approach: assign colors as vertex attributes
        # For proper texture support, you'd use TexturesVertex or TexturesUV

        # Choose rasterization settings
        raster_settings = self.raster_settings_soft if soft else self.raster_settings

        # Create rasterizer
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )

        # Create Phong shader (realistic lighting model)
        # Phong shading considers:
        # - Ambient: base light everywhere
        # - Diffuse: light based on surface angle
        # - Specular: shiny highlights
        if soft:
            shader = SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights
            )
        else:
            shader = HardPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights
            )

        # Complete renderer
        renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=shader
        )

        # Render RGB image
        # Note: For this to work properly with colors, we need proper texture setup
        # For now, this will render with default gray appearance
        images = renderer(mesh)

        # Extract RGB channels
        rgb = images[..., :3]  # (1, H, W, 3)

        return rgb

    def render_depth(self,
                     vertices: torch.Tensor,
                     faces: torch.Tensor,
                     cameras) -> torch.Tensor:
        """
        Render depth map (distance from camera to surface).

        Depth maps are useful for:
        - 3D reconstruction from depth sensors
        - Understanding 3D structure
        - Geometric constraints in optimization

        What is a depth map?
        - Each pixel stores the distance from camera to nearest surface
        - Closer surfaces = smaller depth values
        - Background = large depth values (or infinity)

        Args:
            vertices: (V, 3) vertex positions
            faces: (F, 3) triangle indices
            cameras: PyTorch3D camera object

        Returns:
            depth: (1, H, W, 1) tensor with depth values
                   Small values = close to camera
                   Large values = far from camera

        Example:
            renderer = DifferentiableRenderer()
            depth = renderer.render_depth(vertices, faces, camera)
            # Visualize: plt.imshow(depth[0, :, :, 0])
        """
        # Create mesh
        mesh = Meshes(
            verts=[vertices],
            faces=[faces]
        ).to(self.device)

        # Create rasterizer (use soft settings for depth)
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=self.raster_settings_soft
        )

        # Rasterize to get fragments
        fragments = rasterizer(mesh)

        # Extract depth from fragments
        # fragments.zbuf contains depth values
        # Shape: (1, H, W, faces_per_pixel)
        depth = fragments.zbuf[..., 0:1]  # Take closest face depth

        # Handle background (infinite depth) by clamping
        depth = torch.clamp(depth, 0, 100)  # Clamp to reasonable range

        return depth

    def render_mesh(self,
                    vertices: torch.Tensor,
                    faces: torch.Tensor,
                    cameras,
                    mode: str = 'silhouette',
                    **kwargs) -> torch.Tensor:
        """
        Unified rendering interface - choose rendering mode.

        Args:
            vertices: (V, 3) vertex positions
            faces: (F, 3) triangle indices
            cameras: PyTorch3D camera object
            mode: 'silhouette', 'rgb', or 'depth'
            **kwargs: Additional arguments for specific render modes

        Returns:
            Rendered image tensor
        """
        if mode == 'silhouette':
            return self.render_silhouette(vertices, faces, cameras, **kwargs)
        elif mode == 'rgb':
            return self.render_rgb(vertices, faces, cameras, **kwargs)
        elif mode == 'depth':
            return self.render_depth(vertices, faces, cameras, **kwargs)
        else:
            raise ValueError(f"Unknown render mode: {mode}")


# ==================== UTILITY FUNCTIONS ====================

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
        plt.imsave(path, img_np, cmap='gray')
    else:
        # RGB
        plt.imsave(path, img_np)

    print(f"Saved rendered image to {path}")


def visualize_renders(silhouette: torch.Tensor,
                      rgb: Optional[torch.Tensor] = None,
                      depth: Optional[torch.Tensor] = None,
                      save_path: str = None):
    """
    Visualize different rendering modes side by side.

    Args:
        silhouette: Rendered silhouette
        rgb: Optional rendered RGB image
        depth: Optional rendered depth map
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt

    # Count how many renders we have
    renders = [silhouette]
    titles = ['Silhouette']

    if rgb is not None:
        renders.append(rgb)
        titles.append('RGB')

    if depth is not None:
        renders.append(depth)
        titles.append('Depth')

    # Create subplots
    fig, axes = plt.subplots(1, len(renders), figsize=(6*len(renders), 6))

    if len(renders) == 1:
        axes = [axes]

    for ax, render, title in zip(axes, renders, titles):
        img = render[0].detach().cpu().numpy()

        if img.shape[-1] == 1:
            # Grayscale
            ax.imshow(img.squeeze(-1), cmap='gray')
        else:
            # RGB
            ax.imshow(img)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


# ==================== UNIT TESTS ====================

def test_renderer_initialization():
    """Test renderer creation"""
    renderer = DifferentiableRenderer(image_size=128, device='cpu')

    assert renderer.image_size == 128
    assert renderer.device == 'cpu'

    print("✓ test_renderer_initialization passed")


def test_render_silhouette():
    """Test silhouette rendering"""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from src.mesh import create_icosphere
    from src.camera import create_camera

    # Create simple mesh
    vertices, faces = create_icosphere(subdivisions=0)

    # Create camera
    camera = create_camera(azimuth=0, elevation=30, distance=3.0, device='cpu')

    # Create renderer
    renderer = DifferentiableRenderer(image_size=128, device='cpu')

    # Render
    silhouette = renderer.render_silhouette(vertices, faces, camera)

    # Check output shape
    assert silhouette.shape == (1, 128, 128, 1), \
        f"Expected shape (1, 128, 128, 1), got {silhouette.shape}"

    # Check values are in [0, 1]
    assert silhouette.min() >= 0 and silhouette.max() <= 1, \
        "Silhouette values should be in [0, 1]"

    # Check that some pixels are non-zero (sphere is visible)
    assert silhouette.sum() > 0, "Silhouette should have non-zero pixels"

    print("✓ test_render_silhouette passed")


def test_render_depth():
    """Test depth rendering"""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from src.mesh import create_icosphere
    from src.camera import create_camera

    vertices, faces = create_icosphere(subdivisions=0)
    camera = create_camera(azimuth=0, elevation=30, distance=3.0, device='cpu')
    renderer = DifferentiableRenderer(image_size=128, device='cpu')

    depth = renderer.render_depth(vertices, faces, camera)

    # Check output shape
    assert depth.shape == (1, 128, 128, 1), \
        f"Expected shape (1, 128, 128, 1), got {depth.shape}"

    # Check that depth values are reasonable
    assert depth.min() >= 0, "Depth should be non-negative"

    print("✓ test_render_depth passed")


def test_differentiability():
    """Test that rendering is differentiable (gradients flow)"""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from src.mesh import create_icosphere
    from src.camera import create_camera

    vertices, faces = create_icosphere(subdivisions=0)
    camera = create_camera(azimuth=0, elevation=30, distance=3.0, device='cpu')
    renderer = DifferentiableRenderer(image_size=64, device='cpu')

    # Make vertices require gradients
    vertices_opt = vertices.clone().detach().requires_grad_(True)

    # Render
    silhouette = renderer.render_silhouette(
        vertices_opt, faces, camera, soft=True)

    # Compute a dummy loss
    loss = silhouette.sum()

    # Backpropagate
    loss.backward()

    # Check that gradients exist
    assert vertices_opt.grad is not None, "Gradients should exist"
    assert vertices_opt.grad.abs().sum() > 0, "Gradients should be non-zero"

    print("✓ test_differentiability passed")
    print(f"  Gradient magnitude: {vertices_opt.grad.abs().mean():.6f}")


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))

    print("Testing differentiable renderer...")
    print()

    # Run tests
    test_renderer_initialization()
    test_render_silhouette()
    test_render_depth()
    test_differentiability()

    print()
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print()

    # Demo: Render a sphere from different angles
    print("Creating demo renderings...")
    from src.mesh import create_icosphere, create_deformed_sphere
    from src.camera import create_camera
    import os

    os.makedirs("tests/renders", exist_ok=True)

    # Create mesh
    vertices, faces = create_icosphere(subdivisions=1)

    # Create renderer
    renderer = DifferentiableRenderer(image_size=256, device='cpu')

    # Render from different angles
    angles = [0, 45, 90, 135]

    print("\nRendering sphere from different viewpoints:")
    print("-" * 60)

    for angle in angles:
        camera = create_camera(azimuth=angle, elevation=30,
                               distance=3.0, device='cpu')

        # Render silhouette
        silhouette = renderer.render_silhouette(
            vertices, faces, camera, soft=True)
        save_rendered_image(
            silhouette, f"tests/renders/sphere_silhouette_{angle}deg.png")

        # Render depth
        depth = renderer.render_depth(vertices, faces, camera)
        # Normalize depth for visualization
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        save_rendered_image(
            depth_norm, f"tests/renders/sphere_depth_{angle}deg.png")

        print(f"  Rendered at azimuth={angle}°")

    # Compare sphere vs ellipsoid
    print("\nRendering different shapes:")
    print("-" * 60)

    # Sphere
    v_sphere, f_sphere = create_icosphere(subdivisions=1)
    camera = create_camera(azimuth=45, elevation=30,
                           distance=3.0, device='cpu')
    sil_sphere = renderer.render_silhouette(
        v_sphere, f_sphere, camera, soft=True)
    save_rendered_image(sil_sphere, "tests/renders/shape_sphere.png")
    print("  Rendered sphere")

    # Ellipsoid
    v_ellipsoid, f_ellipsoid = create_deformed_sphere(
        'ellipsoid', {'scale': [1.5, 1.0, 0.8]})
    sil_ellipsoid = renderer.render_silhouette(
        v_ellipsoid, f_ellipsoid, camera, soft=True)
    save_rendered_image(sil_ellipsoid, "tests/renders/shape_ellipsoid.png")
    print("  Rendered ellipsoid")

    # Bump
    v_bump, f_bump = create_deformed_sphere('bump', {'amplitude': 0.3})
    sil_bump = renderer.render_silhouette(v_bump, f_bump, camera, soft=True)
    save_rendered_image(sil_bump, "tests/renders/shape_bump.png")
    print("  Rendered bump")

    print()
    print("=" * 60)
    print("Demo complete! Check tests/renders/ folder")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  • Silhouette rendering = simple binary mask (shape only)")
    print("  • RGB rendering = realistic with lighting and colors")
    print("  • Depth rendering = distance information")
    print("  • All modes are differentiable! (gradients flow to vertices)")
    print("  • Soft rendering = smoother gradients = better optimization")
