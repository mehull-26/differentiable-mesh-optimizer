"""
Mesh Optimization - The Heart of Inverse Rendering!

This module implements gradient-based optimization to reconstruct 3D shapes
from 2D images. This is where all the pieces come together:

mesh + camera + renderer + losses → optimization → reconstructed shape!

The magic of differentiable rendering:
1. Render mesh → get image
2. Compare with target → get loss
3. Backpropagate → get gradients on vertices
4. Update vertices → mesh changes shape
5. Repeat → mesh converges to target shape!
"""

import torch
import torch.optim as optim
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm


class MeshOptimizer:
    """
    Optimizer for mesh reconstruction from images.

    This class manages the entire optimization process:
    - Making mesh vertices optimizable
    - Setting up PyTorch optimizer (Adam, SGD, etc.)
    - Running training loop with backpropagation
    - Logging metrics and saving checkpoints
    - Visualizing progress

    Think of it as a "trainer" for your mesh!
    """

    def __init__(self,
                 renderer,
                 loss_weights: Optional[Dict[str, float]] = None,
                 learning_rate: float = 0.01,
                 optimizer_type: str = 'adam',
                 adaptive_regularization: bool = False,
                 device: str = 'cpu'):
        """
        Initialize the mesh optimizer.

        Args:
            renderer: DifferentiableRenderer instance
            loss_weights: Dict of loss weights (silhouette, edge, laplacian, etc.)
            learning_rate: How big the optimization steps are
                          - Too high: Unstable, may diverge
                          - Too low: Slow convergence
                          - Sweet spot: Usually 0.001 to 0.1
            optimizer_type: 'adam' (adaptive, usually better) or 'sgd' (simple)
            device: 'cpu' or 'cuda'

        Learning rate explained:
            Imagine walking down a hill (gradient descent):
            - LR = 0.001: Baby steps (safe but slow)
            - LR = 0.01:  Normal walking (good balance)
            - LR = 0.1:   Running (fast but might overshoot)
            - LR = 1.0:   Sprinting (likely to fall/diverge!)
        """
        self.renderer = renderer
        self.device = device

        # Default loss weights
        if loss_weights is None:
            self.loss_weights = {
                'silhouette': 1.0,    # Primary data term
                'edge': 0.1,          # Edge regularization
                'laplacian': 0.1,     # Smoothness
                'normal': 0.01        # Normal consistency
            }
        else:
            self.loss_weights = loss_weights

        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.adaptive_regularization = adaptive_regularization

        # Will be set during optimization
        self.optimizer = None
        self.vertices = None
        self.faces = None
        self.adaptive_loss_weights = None  # Will be computed if adaptive enabled

        # Tracking
        self.history = {
            'total_loss': [],
            'silhouette_loss': [],
            'edge_loss': [],
            'laplacian_loss': [],
            'normal_loss': []
        }

        print(f"Initialized MeshOptimizer:")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Optimizer: {optimizer_type}")
        print(f"  - Loss weights: {self.loss_weights}")

    def setup_optimization(self,
                           initial_vertices: torch.Tensor,
                           faces: torch.Tensor) -> torch.Tensor:
        """
        Prepare mesh for optimization.

        This is a CRITICAL step!

        Why do we need this?
        - By default, tensors don't track gradients
        - We need to tell PyTorch: "I want to optimize these vertices!"
        - .requires_grad_(True) enables gradient computation

        What happens behind the scenes:
        1. Clone vertices to avoid modifying original
        2. Enable gradient tracking
        3. Create optimizer that will update these vertices

        Args:
            initial_vertices: (V, 3) starting vertex positions
            faces: (F, 3) triangle indices (fixed, not optimized)

        Returns:
            optimizable_vertices: Vertices with requires_grad=True
        """
        # Clone and move to device
        self.vertices = initial_vertices.clone().to(self.device)
        self.faces = faces.to(self.device)

        # THIS IS THE MAGIC LINE!
        # Enable gradient computation for vertices
        self.vertices.requires_grad_(True)

        # Create PyTorch optimizer
        # The optimizer will modify self.vertices based on gradients
        if self.optimizer_type.lower() == 'adam':
            # Adam: Adaptive learning rates, momentum
            # Usually the best choice for mesh optimization
            self.optimizer = optim.Adam([self.vertices], lr=self.learning_rate)
        elif self.optimizer_type.lower() == 'sgd':
            # SGD: Simple gradient descent
            # More predictable but often slower
            self.optimizer = optim.SGD([self.vertices], lr=self.learning_rate,
                                       momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

        print(f"\n✓ Optimization setup complete")
        print(f"  - Vertices: {self.vertices.shape}")
        print(f"  - Faces: {self.faces.shape}")
        print(f"  - Requires grad: {self.vertices.requires_grad}")

        # Compute adaptive regularization weights if enabled
        if self.adaptive_regularization:
            self._compute_adaptive_weights()

        return self.vertices

    def _compute_adaptive_weights(self):
        """
        Compute adaptive regularization weights based on mesh complexity.

        Higher subdivision levels have more vertices/edges, so we need
        stronger regularization to maintain smoothness.

        Scaling factor based on number of vertices:
        - Level 0 (12 verts): factor = 1.0 (base)
        - Level 1 (42 verts): factor ≈ 1.88
        - Level 2 (162 verts): factor ≈ 3.67
        - Level 3 (642 verts): factor ≈ 7.32
        - Level 4 (2562 verts): factor ≈ 14.6
        """
        num_verts = self.vertices.shape[0]
        base_verts = 12  # Level 0 icosphere

        # Compute scaling factor: sqrt(num_verts / base_verts)
        # Using sqrt to avoid over-regularization at high levels
        scale_factor = torch.sqrt(torch.tensor(num_verts / base_verts)).item()

        # Apply scaling to regularization terms (not to data term)
        self.adaptive_loss_weights = {
            # Keep data term unchanged
            'silhouette': self.loss_weights['silhouette'],
            'edge': self.loss_weights['edge'] * scale_factor,
            'laplacian': self.loss_weights['laplacian'] * scale_factor,
            'normal': self.loss_weights['normal'] * scale_factor
        }

        print(f"  - Adaptive regularization enabled:")
        print(f"    Vertices: {num_verts}, Scale factor: {scale_factor:.2f}")
        print(
            f"    Edge weight: {self.loss_weights['edge']:.3f} → {self.adaptive_loss_weights['edge']:.3f}")
        print(
            f"    Laplacian weight: {self.loss_weights['laplacian']:.3f} → {self.adaptive_loss_weights['laplacian']:.3f}")
        print(
            f"    Normal weight: {self.loss_weights['normal']:.4f} → {self.adaptive_loss_weights['normal']:.4f}")

    def optimization_step(self,
                          target_images: List[torch.Tensor],
                          cameras: list) -> Dict[str, float]:
        """
        Perform one optimization iteration.

        This is the CORE of gradient-based optimization!

        The optimization cycle:
        1. Zero gradients (clear previous iteration)
        2. Render mesh from all cameras
        3. Compute loss vs targets
        4. Backpropagate (.backward() - compute gradients)
        5. Update vertices (.step() - apply gradients)

        Args:
            target_images: List of target silhouettes, one per camera
            cameras: List of camera objects

        Returns:
            loss_dict: Dictionary with all loss components
        """
        # Import losses here to avoid circular imports
        from src.losses import combined_loss

        # Step 1: Zero gradients
        # Why? Gradients accumulate by default in PyTorch
        # We want fresh gradients for this iteration
        self.optimizer.zero_grad()

        # Step 2: Render from all views
        total_silhouette_loss = 0.0
        num_views = len(cameras)

        for camera, target in zip(cameras, target_images):
            # Render current mesh
            rendered = self.renderer.render_silhouette(
                self.vertices,
                self.faces,
                camera
            )

            # Accumulate silhouette loss across views
            target = target.to(self.device)
            total_silhouette_loss += torch.nn.functional.mse_loss(
                rendered, target)

        # Average over views
        avg_silhouette_loss = total_silhouette_loss / num_views

        # Step 3: Compute regularization losses
        from src.losses import edge_loss, laplacian_loss, normal_consistency_loss

        loss_edge = edge_loss(self.vertices, self.faces)
        loss_lap = laplacian_loss(self.vertices, self.faces)
        loss_normal = normal_consistency_loss(self.vertices, self.faces)

        # Use adaptive weights if enabled, otherwise use base weights
        weights = self.adaptive_loss_weights if self.adaptive_loss_weights is not None else self.loss_weights

        # Step 4: Combine all losses
        total_loss = (
            weights['silhouette'] * avg_silhouette_loss +
            weights['edge'] * loss_edge +
            weights['laplacian'] * loss_lap +
            weights['normal'] * loss_normal
        )

        # Step 5: Backpropagate
        # THIS IS WHERE THE MAGIC HAPPENS!
        # .backward() computes gradients: ∂loss/∂vertices
        total_loss.backward()

        # Step 6: Update vertices
        # optimizer.step() does: vertices = vertices - lr * gradients
        self.optimizer.step()

        # Return losses for logging
        loss_dict = {
            'total': total_loss.item(),
            'silhouette': avg_silhouette_loss.item(),
            'edge': loss_edge.item(),
            'laplacian': loss_lap.item(),
            'normal': loss_normal.item()
        }

        return loss_dict

    def optimize_mesh(self,
                      initial_vertices: torch.Tensor,
                      faces: torch.Tensor,
                      target_images: List[torch.Tensor],
                      cameras: list,
                      num_iterations: int = 200,
                      log_interval: int = 10,
                      checkpoint_interval: int = 50,
                      checkpoint_mode: str = "iteration",
                      save_checkpoint_renders: bool = True,
                      checkpoint_render_views: int = 4,
                      verbose: bool = True,
                      output_dir: Optional[str] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Main optimization loop - THE HEART OF INVERSE RENDERING!

        This function brings everything together:
        mesh + camera + renderer + losses → optimization → reconstructed shape

        The training loop:
        ```
        for iteration in range(num_iterations):
            1. Render current mesh
            2. Compute loss vs target
            3. Backpropagate gradients
            4. Update vertices
            5. Log progress
            6. Save checkpoints
        ```

        Args:
            initial_vertices: Starting mesh vertices
            faces: Triangle faces (fixed)
            target_images: List of target silhouettes
            cameras: List of cameras for each target
            num_iterations: How many optimization steps
            log_interval: Print progress every N iterations
            checkpoint_interval: Save checkpoint value (interpretation depends on mode)
            checkpoint_mode: "iteration" (every N iters) or "percentage" (at N% intervals)
            save_checkpoint_renders: Whether to save multi-angle renders
            checkpoint_render_views: Number of viewing angles for renders
            verbose: Control detailed console output
            output_dir: Where to save outputs

        Returns:
            optimized_vertices: Final optimized mesh
            history: Dict with loss history

        Learning checkpoint items:
        - Print gradients after first iteration ✓
        - Understand what .backward() does ✓
        - See how vertices change over iterations ✓
        - Visualize loss decreasing ✓
        """
        if verbose:
            print("\n" + "="*70)
            print("STARTING MESH OPTIMIZATION")
            print("="*70)

        # Setup
        self.setup_optimization(initial_vertices, faces)

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        # Store initial state for comparison
        initial_vertices_copy = self.vertices.detach().clone()

        # Calculate checkpoint iterations based on mode
        if checkpoint_mode == "percentage":
            # checkpoint_interval represents percentage (e.g., 25 means save at 0%, 25%, 50%, 75%, 100%)
            checkpoint_iterations = set()
            for pct in range(0, 101, checkpoint_interval):
                iter_num = int((pct / 100.0) * num_iterations)
                if iter_num > 0:  # Skip 0 as it's saved separately
                    checkpoint_iterations.add(iter_num)
            checkpoint_iterations.add(num_iterations)  # Always save final
            if verbose:
                print(
                    f"  Checkpoint mode: percentage-based ({checkpoint_interval}% intervals)")
                print(
                    f"  Checkpoint iterations: {sorted(checkpoint_iterations)}")
        else:  # "iteration" mode
            # checkpoint_interval represents iteration count (e.g., 50 means every 50 iterations)
            checkpoint_iterations = set(
                range(checkpoint_interval, num_iterations + 1, checkpoint_interval))
            checkpoint_iterations.add(num_iterations)  # Always save final
            if verbose:
                print(
                    f"  Checkpoint mode: iteration-based (every {checkpoint_interval} iterations)")

        # Optimization loop with progress bar
        pbar = tqdm(range(num_iterations),
                    desc="Optimizing", disable=not verbose)

        for iteration in pbar:
            # Perform one optimization step
            loss_dict = self.optimization_step(target_images, cameras)

            # Record history
            self.history['total_loss'].append(loss_dict['total'])
            self.history['silhouette_loss'].append(loss_dict['silhouette'])
            self.history['edge_loss'].append(loss_dict['edge'])
            self.history['laplacian_loss'].append(loss_dict['laplacian'])
            self.history['normal_loss'].append(loss_dict['normal'])

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'sil': f"{loss_dict['silhouette']:.4f}"
            })

            # LEARNING CHECKPOINT: Print gradients after first iteration
            if iteration == 0 and verbose:
                print("\n" + "="*70)
                print("LEARNING CHECKPOINT: First Iteration")
                print("="*70)
                print(f"Loss: {loss_dict['total']:.6f}")
                print(f"Gradient stats:")
                print(f"  - Mean: {self.vertices.grad.mean().item():.6e}")
                print(f"  - Std:  {self.vertices.grad.std().item():.6e}")
                print(f"  - Max:  {self.vertices.grad.abs().max().item():.6e}")
                print("\nWhat .backward() did:")
                print("  1. Computed ∂loss/∂vertices for every vertex")
                print("  2. Stored gradients in vertices.grad")
                print("  3. These gradients tell us how to change vertices")
                print("  4. Positive gradient → increase vertex coordinate")
                print("  5. Negative gradient → decrease vertex coordinate")
                print("="*70 + "\n")

            # Log progress
            if verbose and (iteration + 1) % log_interval == 0:
                # Compute vertex displacement
                displacement = (
                    self.vertices - initial_vertices_copy).norm(dim=1).mean()

                print(f"\nIteration {iteration + 1}/{num_iterations}")
                print(f"  Total loss:      {loss_dict['total']:.6f}")
                print(f"  Silhouette loss: {loss_dict['silhouette']:.6f}")
                print(f"  Edge loss:       {loss_dict['edge']:.6f}")
                print(f"  Laplacian loss:  {loss_dict['laplacian']:.6f}")
                print(f"  Avg displacement: {displacement.item():.6f}")

            # Save checkpoint with consistent naming: checkpoint_IIII_PP
            # IIII = iteration number (0000, 0010, 0025, etc.)
            # PP = percentage complete (00, 10, 25, 50, 75, 100)
            if output_dir and (iteration + 1) in checkpoint_iterations:
                percent_complete = int(
                    ((iteration + 1) / num_iterations) * 100)
                checkpoint_name = f"checkpoint_{iteration+1:04d}_{percent_complete:02d}"

                self.save_checkpoint(
                    iteration + 1,
                    output_path / f"{checkpoint_name}.obj",
                    verbose=verbose
                )
                # Also save multi-angle renders if enabled
                if save_checkpoint_renders:
                    self.save_checkpoint_renders(
                        iteration + 1,
                        output_path,
                        cameras=cameras,  # Pass actual cameras from optimization
                        num_views=checkpoint_render_views,
                        image_size=256,
                        filename=f"{checkpoint_name}.png",
                        verbose=verbose
                    )

        if verbose:
            print("\n" + "="*70)
            print("OPTIMIZATION COMPLETE")
            print("="*70)

            # Final statistics
            final_displacement = (
                self.vertices - initial_vertices_copy).norm(dim=1)
            print(f"\nFinal statistics:")
            print(
                f"  - Mean vertex displacement: {final_displacement.mean().item():.6f}")
            print(
                f"  - Max vertex displacement:  {final_displacement.max().item():.6f}")
            print(
                f"  - Final total loss:         {self.history['total_loss'][-1]:.6f}")
            print(
                f"  - Initial total loss:       {self.history['total_loss'][0]:.6f}")
            print(
                f"  - Loss reduction:           {(1 - self.history['total_loss'][-1]/self.history['total_loss'][0])*100:.1f}%")

        # Save final results with consistent naming
        if output_dir:
            # Save final mesh with consistent naming: checkpoint_IIII_100
            final_checkpoint_name = f"checkpoint_{num_iterations:04d}_100"
            final_mesh_path = output_path / f"{final_checkpoint_name}.obj"
            self.save_checkpoint(
                num_iterations, final_mesh_path, verbose=verbose)

            # Save final multi-angle renders if enabled
            if save_checkpoint_renders:
                self.save_checkpoint_renders(
                    num_iterations,
                    output_path,
                    cameras=cameras,  # Use actual cameras for consistency
                    num_views=checkpoint_render_views,
                    image_size=256,
                    filename=f"{final_checkpoint_name}.png",
                    verbose=verbose
                )
            if verbose:
                self.save_loss_plot(output_path / "loss_curve.png")
                self.save_history(output_path / "optimization_history.json")
            else:
                self.save_loss_plot(
                    output_path / "loss_curve.png", verbose=False)
                self.save_history(
                    output_path / "optimization_history.json", verbose=False)

        return self.vertices.detach(), self.history

    def save_checkpoint(self, iteration: int, path: Path, verbose: bool = True):
        """Save current mesh to file"""
        from src.mesh import save_mesh
        save_mesh(
            self.vertices.detach().cpu(),
            self.faces.cpu(),
            str(path)
        )
        if verbose:
            print(f"  Saved checkpoint: {path}")

    def save_checkpoint_renders(self, iteration: int, output_dir: Path,
                                cameras: list = None,
                                num_views: int = 4, image_size: int = 256,
                                filename: str = None, verbose: bool = True):
        """
        Save rendered views of current mesh from same viewpoints as targets.

        This creates a grid of renders showing the mesh from the same camera
        viewpoints used during optimization, making it easy to compare with
        ground truth target images.

        Args:
            iteration: Current iteration number
            output_dir: Directory to save renders
            cameras: List of camera objects (if None, creates rotating views)
            num_views: Number of viewing angles (used only if cameras=None)
            image_size: Size of each rendered view (used only if cameras=None)
            filename: Custom filename (default: iteration_XXXX.png)
        """
        from src.camera import create_camera
        from src.renderer import save_rendered_image

        # Create subdirectory for checkpoint renders
        renders_dir = output_dir / "checkpoint_renders"
        renders_dir.mkdir(exist_ok=True)

        if cameras is not None:
            # Use the actual cameras from optimization
            num_views = len(cameras)
            fig, axes = plt.subplots(1, num_views, figsize=(4*num_views, 4))
            if num_views == 1:
                axes = [axes]

            fig.suptitle(f'Iteration {iteration}',
                         fontsize=16, fontweight='bold')

            for idx, camera in enumerate(cameras):
                # Render from this camera viewpoint
                rendered = self.renderer.render_silhouette(
                    self.vertices,
                    self.faces,
                    camera
                )

                # Display in subplot (detach to avoid grad issues)
                axes[idx].imshow(
                    rendered[0, :, :, 0].detach().cpu().numpy(), cmap='gray')
                axes[idx].set_title(f'View {idx}', fontsize=12)
                axes[idx].axis('off')
        else:
            # Fallback: create rotating views
            azimuths = [i * (360.0 / num_views) for i in range(num_views)]
            elevation = 30.0
            distance = 3.0

            fig, axes = plt.subplots(1, num_views, figsize=(4*num_views, 4))
            if num_views == 1:
                axes = [axes]

            fig.suptitle(f'Iteration {iteration}',
                         fontsize=16, fontweight='bold')

            for idx, azimuth in enumerate(azimuths):
                # Create camera for this angle
                camera = create_camera(
                    azimuth=azimuth,
                    elevation=elevation,
                    distance=distance,
                    image_size=image_size,
                    device=self.device
                )

                # Render from this angle
                rendered = self.renderer.render_silhouette(
                    self.vertices,
                    self.faces,
                    camera
                )

                # Display in subplot (detach to avoid grad issues)
                axes[idx].imshow(
                    rendered[0, :, :, 0].detach().cpu().numpy(), cmap='gray')
                axes[idx].set_title(f'Azimuth {azimuth:.0f}°', fontsize=12)
                axes[idx].axis('off')

        plt.tight_layout()

        # Save the grid with custom or default filename
        if filename is None:
            filename = f"checkpoint_{iteration:04d}.png"
        output_path = renders_dir / filename
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        if verbose:
            print(f"  Saved checkpoint renders: {output_path}")

    def save_loss_plot(self, path: Path, verbose: bool = True):
        """Create and save loss curve visualization"""
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Plot total loss
        axes[0].plot(self.history['total_loss'], linewidth=2)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Total Loss Over Time')
        axes[0].grid(True, alpha=0.3)

        # Plot individual loss components
        axes[1].plot(self.history['silhouette_loss'],
                     label='Silhouette', linewidth=2)
        axes[1].plot(self.history['edge_loss'], label='Edge', linewidth=2)
        axes[1].plot(self.history['laplacian_loss'],
                     label='Laplacian', linewidth=2)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Individual Loss Components')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

        if verbose:
            print(f"  Saved loss plot: {path}")

    def save_history(self, path: Path, verbose: bool = True):
        """Save optimization history to JSON"""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        if verbose:
            print(f"  Saved history: {path}")


def optimize_mesh(initial_vertices: torch.Tensor,
                  faces: torch.Tensor,
                  target_images: List[torch.Tensor],
                  cameras: list,
                  config: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
    """
    Convenience function for mesh optimization.

    This is a simpler interface to MeshOptimizer for quick use.

    Args:
        initial_vertices: Starting mesh
        faces: Triangle faces
        target_images: Target silhouettes
        cameras: Camera list
        config: Optional configuration dict with keys:
               - learning_rate
               - num_iterations
               - loss_weights
               - device
               - output_dir

    Returns:
        optimized_vertices: Final mesh
        history: Loss history
    """
    if config is None:
        config = {}

    # Import renderer
    from src.renderer import DifferentiableRenderer

    # Create renderer
    device = config.get('device', 'cpu')
    image_size = target_images[0].shape[1] if target_images else 256
    renderer = DifferentiableRenderer(image_size=image_size, device=device)

    # Create optimizer
    optimizer = MeshOptimizer(
        renderer=renderer,
        loss_weights=config.get('loss_weights'),
        learning_rate=config.get('learning_rate', 0.01),
        optimizer_type=config.get('optimizer_type', 'adam'),
        device=device
    )

    # Run optimization
    optimized_vertices, history = optimizer.optimize_mesh(
        initial_vertices=initial_vertices,
        faces=faces,
        target_images=target_images,
        cameras=cameras,
        num_iterations=config.get('num_iterations', 200),
        log_interval=config.get('log_interval', 10),
        checkpoint_interval=config.get('checkpoint_interval', 50),
        output_dir=config.get('output_dir')
    )

    return optimized_vertices, history


# ==================== UNIT TESTS ====================

def test_optimizer_setup():
    """Test optimizer initialization and setup"""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from src.renderer import DifferentiableRenderer
    from src.mesh import create_icosphere

    renderer = DifferentiableRenderer(image_size=64, device='cpu')
    optimizer = MeshOptimizer(renderer, learning_rate=0.01, device='cpu')

    vertices, faces = create_icosphere(subdivisions=1)
    opt_vertices = optimizer.setup_optimization(vertices, faces)

    assert opt_vertices.requires_grad, "Vertices should require gradients"
    assert optimizer.optimizer is not None, "Optimizer should be initialized"

    print("✓ test_optimizer_setup passed")


def test_optimization_step():
    """Test single optimization step"""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from src.renderer import DifferentiableRenderer
    from src.mesh import create_icosphere
    from src.camera import create_camera

    # Setup
    renderer = DifferentiableRenderer(image_size=64, device='cpu')
    optimizer_obj = MeshOptimizer(renderer, learning_rate=0.01, device='cpu')

    vertices, faces = create_icosphere(subdivisions=1)
    optimizer_obj.setup_optimization(vertices, faces)

    # Create dummy target
    camera = create_camera(0, 30, 3.0, image_size=64, device='cpu')
    target = torch.ones(1, 64, 64, 1)

    # Run one step
    loss_dict = optimizer_obj.optimization_step([target], [camera])

    assert 'total' in loss_dict, "Should return total loss"
    assert 'silhouette' in loss_dict, "Should return silhouette loss"
    assert optimizer_obj.vertices.grad is not None, "Should have gradients"

    print("✓ test_optimization_step passed")
    print(f"  Loss: {loss_dict['total']:.6f}")


if __name__ == "__main__":
    print("Testing mesh optimizer...")
    print()

    test_optimizer_setup()
    print()

    test_optimization_step()
    print()

    print("=" * 60)
    print("All optimizer tests passed! ✓")
    print("=" * 60)
    print()
    print("Key concepts demonstrated:")
    print("  • Vertices with requires_grad=True enable optimization")
    print("  • .backward() computes gradients ∂loss/∂vertices")
    print("  • .step() updates vertices using gradients")
    print("  • Gradients tell us how to change shape to reduce loss")
    print()
    print("Optimizer comparison:")
    print("  • Adam: Adaptive learning rates, momentum")
    print("         Usually best for mesh optimization")
    print("  • SGD:  Simple gradient descent")
    print("         More predictable, may need tuning")
    print()
    print("Learning rate effects:")
    print("  • Too low (0.0001):  Slow, stable")
    print("  • Good (0.01):       Fast, stable")
    print("  • Too high (0.5):    Fast but unstable")
    print("  • Way too high (5):  Diverges!")
