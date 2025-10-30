"""
Loss Functions for 3D Reconstruction

This module implements various loss functions used to optimize mesh parameters.
Understanding losses is crucial for successful 3D reconstruction!

Why do we need different losses?
1. Data loss: Match the rendered output to target images
2. Regularization: Keep the mesh well-behaved (smooth, no degenerate triangles)

Think of it like this:
- Data loss says: "Make the shape look like the target"
- Regularization says: "But don't create a crazy, jagged mesh in the process"
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def silhouette_loss(rendered: torch.Tensor,
                    target: torch.Tensor,
                    loss_type: str = 'l2') -> torch.Tensor:
    """
    Compute loss between rendered and target silhouettes.

    This is the PRIMARY loss for shape reconstruction from silhouettes!

    What is a silhouette?
    - Binary mask: 1 where object is visible, 0 for background
    - Captures the shape/outline of the object
    - Ignores color, texture, lighting (just shape!)

    Why silhouette loss works:
    - Simple: Only need to match object boundaries
    - Effective: Strong signal for shape optimization
    - Fast: No need to model lighting or colors

    Args:
        rendered: (B, H, W, 1) rendered silhouette from current mesh
        target: (B, H, W, 1) target silhouette we want to match
        loss_type: 'l2' (MSE) or 'bce' (Binary Cross-Entropy)

    Returns:
        loss: Scalar tensor, lower = better match

    Example:
        >>> rendered = renderer.render_silhouette(vertices, faces, camera)
        >>> target = load_target_image("target.png")
        >>> loss = silhouette_loss(rendered, target)
        >>> loss.backward()  # Compute gradients
        >>> # Now vertices.grad tells us how to change the mesh!
    """
    if loss_type == 'l2':
        # L2 loss (Mean Squared Error)
        # Simple and effective: (rendered - target)^2
        # Good for: Soft silhouettes with smooth edges
        loss = F.mse_loss(rendered, target)

    elif loss_type == 'bce':
        # Binary Cross-Entropy
        # Better for: Hard binary masks (0 or 1)
        # More principled for probability distributions
        loss = F.binary_cross_entropy(rendered, target)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use 'l2' or 'bce'")

    return loss


def rgb_loss(rendered: torch.Tensor,
             target: torch.Tensor,
             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute loss between rendered and target RGB images.

    Used when you have color images, not just silhouettes.
    More challenging than silhouette loss because you need to model:
    - Lighting
    - Materials
    - Colors/textures

    Args:
        rendered: (B, H, W, 3) rendered RGB image
        target: (B, H, W, 3) target RGB image
        mask: Optional (B, H, W, 1) mask to ignore background

    Returns:
        loss: Scalar tensor

    Why use a mask?
    - We only care about the object, not background
    - Background pixels are arbitrary (lighting, etc.)
    - Mask focuses loss on the object region
    """
    if mask is not None:
        # Only compute loss where mask is 1 (object region)
        # This prevents background from affecting the loss
        rendered_masked = rendered * mask
        target_masked = target * mask

        # Normalize by number of object pixels
        num_pixels = mask.sum() + 1e-8  # Add epsilon to avoid division by zero
        loss = ((rendered_masked - target_masked) ** 2).sum() / num_pixels
    else:
        # Standard L2 loss over entire image
        loss = F.mse_loss(rendered, target)

    return loss


def edge_loss(vertices: torch.Tensor,
              faces: torch.Tensor,
              target_length: Optional[float] = None) -> torch.Tensor:
    """
    Edge length regularization loss.

    Why do we need this?
    - Without regularization, optimization might create:
      * Very long edges (stretched triangles)
      * Very short edges (collapsed vertices)
      * Irregular, jagged meshes

    What this loss does:
    - Penalizes edges that are too long or too short
    - Encourages uniform edge lengths
    - Keeps mesh smooth and well-formed

    Think of it like:
    "I want the shape to match, but also want edges to stay a reasonable size"

    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) triangle indices
        target_length: Optional target edge length (if None, uses initial length)

    Returns:
        loss: Scalar tensor, penalizes deviation from target edge length

    Example scenario without edge loss:
        Initial sphere: All edges ≈ 0.5 units
        After optimization: Some edges → 5.0 units (stretched!)
                           Some edges → 0.01 units (collapsed!)
        Result: Jagged, ugly mesh

    With edge loss:
        Optimization is constrained to keep edges reasonable
        Result: Smooth, clean mesh
    """
    # Get all edges in the mesh
    # Each triangle has 3 edges: (v0,v1), (v1,v2), (v2,v0)
    edges = torch.cat([
        faces[:, [0, 1]],  # Edge from vertex 0 to vertex 1
        faces[:, [1, 2]],  # Edge from vertex 1 to vertex 2
        faces[:, [2, 0]]   # Edge from vertex 2 to vertex 0
    ], dim=0)

    # Get vertex positions for each edge endpoint
    v0 = vertices[edges[:, 0]]  # Start vertices
    v1 = vertices[edges[:, 1]]  # End vertices

    # Compute edge lengths
    edge_vectors = v1 - v0
    edge_lengths = torch.norm(edge_vectors, dim=1)

    if target_length is None:
        # Use mean edge length as target
        target_length = edge_lengths.mean().detach()

    # Penalize deviation from target length
    # Loss = mean((length - target)^2)
    loss = ((edge_lengths - target_length) ** 2).mean()

    return loss


def laplacian_loss(vertices: torch.Tensor,
                   faces: torch.Tensor) -> torch.Tensor:
    """
    Laplacian smoothing regularization.

    This is a MORE SOPHISTICATED regularization than edge loss!

    What is the Laplacian?
    - For each vertex, look at its neighbors
    - The Laplacian is: vertex_position - average_of_neighbors
    - If Laplacian = 0, vertex is at the average of neighbors (smooth!)
    - If Laplacian ≠ 0, vertex is off-center (creates curvature)

    Why is this useful?
    - Prevents spiky, sharp features
    - Encourages smooth surfaces
    - Penalizes degenerate triangles (flat, collapsed)

    Visual analogy:
    Imagine a trampoline with weights at vertices:
    - Laplacian = 0: Each weight at center of its neighbors (flat surface)
    - Laplacian ≠ 0: Weight pushed up or down (creates bumps)

    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) triangle indices

    Returns:
        loss: Scalar tensor, lower = smoother mesh

    Comparison with edge_loss:
    - edge_loss: "Keep edge lengths uniform"
    - laplacian_loss: "Keep surface smooth, prevent spikes"
    - Usually use BOTH together!
    """
    # Build neighbor list for each vertex
    num_vertices = vertices.shape[0]

    # Initialize neighbor tracking
    # neighbor_sum[i] = sum of positions of vertices connected to vertex i
    # neighbor_count[i] = number of neighbors of vertex i
    neighbor_sum = torch.zeros_like(vertices)
    neighbor_count = torch.zeros(
        num_vertices, dtype=torch.float32, device=vertices.device)

    # Each face contributes 3 edges to the neighbor graph
    for i in range(3):
        # For edge (v_i, v_next)
        v_curr = faces[:, i]
        v_next = faces[:, (i + 1) % 3]

        # Add v_next as neighbor of v_curr
        neighbor_sum.index_add_(0, v_curr, vertices[v_next])
        neighbor_count.index_add_(
            0, v_curr, torch.ones_like(v_curr, dtype=torch.float32))

        # Add v_curr as neighbor of v_next (undirected graph)
        neighbor_sum.index_add_(0, v_next, vertices[v_curr])
        neighbor_count.index_add_(
            0, v_next, torch.ones_like(v_next, dtype=torch.float32))

    # Compute average neighbor position
    # Add epsilon to avoid division by zero (isolated vertices)
    neighbor_count = neighbor_count.unsqueeze(1)  # (V, 1)
    neighbor_avg = neighbor_sum / (neighbor_count + 1e-8)

    # Laplacian = vertex - average_of_neighbors
    laplacian = vertices - neighbor_avg

    # Loss = squared norm of Laplacian
    # Small Laplacian = vertex close to neighbor average = smooth
    # Large Laplacian = vertex far from neighbors = not smooth
    loss = (laplacian ** 2).sum(dim=1).mean()

    return loss


def normal_consistency_loss(vertices: torch.Tensor,
                            faces: torch.Tensor) -> torch.Tensor:
    """
    Normal consistency regularization.

    Encourages neighboring triangles to have similar normals.
    This prevents sharp edges and creates smooth surfaces.

    What are normals?
    - Each triangle has a normal vector (perpendicular to surface)
    - Consistent normals = smooth surface
    - Inconsistent normals = sharp edges/creases

    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) triangle indices

    Returns:
        loss: Scalar tensor, penalizes inconsistent normals
    """
    # Get vertices for each face
    v0 = vertices[faces[:, 0]]  # (F, 3)
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Compute face normals using cross product
    # Normal = (v1 - v0) × (v2 - v0)
    edge1 = v1 - v0
    edge2 = v2 - v0
    normals = torch.cross(edge1, edge2, dim=1)

    # Normalize to unit vectors
    normals = F.normalize(normals, dim=1)

    # For each edge, find adjacent faces and compare their normals
    # This is simplified - a full implementation would build edge-face adjacency
    # For now, we encourage all normals to be similar (smooth surface)

    # Compute variance of normals as a proxy for consistency
    # Low variance = consistent normals = smooth surface
    normal_mean = normals.mean(dim=0, keepdim=True)
    loss = ((normals - normal_mean) ** 2).sum(dim=1).mean()

    return loss


def combined_loss(rendered: torch.Tensor,
                  target: torch.Tensor,
                  vertices: torch.Tensor,
                  faces: torch.Tensor,
                  weights: Optional[dict] = None) -> Tuple[torch.Tensor, dict]:
    """
    Combine multiple loss terms with weights.

    Why combine losses?
    - Data loss alone: Matches target but creates ugly mesh
    - Regularization alone: Smooth mesh but doesn't match target
    - Combined: Best of both worlds!

    Think of it as a balance:
    - High data loss weight: Prioritize matching target (may be jagged)
    - High regularization weight: Prioritize smooth mesh (may not match perfectly)

    Args:
        rendered: Rendered silhouette
        target: Target silhouette
        vertices: Vertex positions
        faces: Triangle indices
        weights: Dict with keys: 'silhouette', 'edge', 'laplacian', 'normal'

    Returns:
        total_loss: Combined weighted loss
        loss_dict: Individual loss components for monitoring
    """
    if weights is None:
        # Default weights (balanced)
        weights = {
            'silhouette': 1.0,     # Main data term
            'edge': 0.1,           # Weak edge regularization
            'laplacian': 0.1,      # Weak smoothness
            'normal': 0.01         # Very weak normal consistency
        }

    # Compute individual losses
    loss_sil = silhouette_loss(rendered, target)
    loss_edge = edge_loss(vertices, faces)
    loss_lap = laplacian_loss(vertices, faces)
    loss_normal = normal_consistency_loss(vertices, faces)

    # Weighted combination
    total = (
        weights.get('silhouette', 1.0) * loss_sil +
        weights.get('edge', 0.1) * loss_edge +
        weights.get('laplacian', 0.1) * loss_lap +
        weights.get('normal', 0.01) * loss_normal
    )

    # Return both total and individual losses for monitoring
    loss_dict = {
        'total': total.item(),
        'silhouette': loss_sil.item(),
        'edge': loss_edge.item(),
        'laplacian': loss_lap.item(),
        'normal': loss_normal.item()
    }

    return total, loss_dict


# ==================== UNIT TESTS ====================

def test_silhouette_loss():
    """Test silhouette loss computation"""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))

    # Create dummy silhouettes
    rendered = torch.ones(1, 64, 64, 1) * 0.8  # Mostly white
    target = torch.ones(1, 64, 64, 1) * 1.0    # All white

    # Test L2 loss
    loss_l2 = silhouette_loss(rendered, target, loss_type='l2')
    assert loss_l2.item() > 0, "Loss should be positive when images don't match"

    # Test perfect match
    loss_zero = silhouette_loss(target, target, loss_type='l2')
    assert loss_zero.item() < 1e-6, "Loss should be near zero for identical images"

    print("✓ test_silhouette_loss passed")
    print(f"  L2 loss (mismatch): {loss_l2.item():.6f}")
    print(f"  L2 loss (perfect match): {loss_zero.item():.6f}")


def test_edge_loss():
    """Test edge loss computation"""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from src.mesh import create_icosphere

    vertices, faces = create_icosphere(subdivisions=0)

    # Test with normal sphere
    loss1 = edge_loss(vertices, faces)

    # Test with stretched sphere (edges should be longer)
    vertices_stretched = vertices.clone()
    vertices_stretched[:, 0] *= 2.0  # Stretch along X
    loss2 = edge_loss(vertices_stretched, faces)

    # Stretched sphere should have higher loss
    assert loss2 > loss1, "Stretched sphere should have higher edge loss"

    print("✓ test_edge_loss passed")
    print(f"  Normal sphere: {loss1.item():.6f}")
    print(f"  Stretched sphere: {loss2.item():.6f}")


def test_laplacian_loss():
    """Test Laplacian smoothness loss"""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from src.mesh import create_icosphere

    vertices, faces = create_icosphere(subdivisions=1)

    # Test with smooth sphere
    loss_smooth = laplacian_loss(vertices, faces)

    # Test with perturbed vertices (add noise)
    vertices_noisy = vertices + torch.randn_like(vertices) * 0.1
    loss_noisy = laplacian_loss(vertices_noisy, faces)

    # Noisy mesh should have higher Laplacian loss
    assert loss_noisy > loss_smooth, "Noisy mesh should have higher Laplacian loss"

    print("✓ test_laplacian_loss passed")
    print(f"  Smooth sphere: {loss_smooth.item():.6f}")
    print(f"  Noisy sphere: {loss_noisy.item():.6f}")


def test_combined_loss():
    """Test combined loss with all terms"""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from src.mesh import create_icosphere

    vertices, faces = create_icosphere(subdivisions=1)
    rendered = torch.ones(1, 64, 64, 1) * 0.8
    target = torch.ones(1, 64, 64, 1) * 1.0

    total_loss, loss_dict = combined_loss(rendered, target, vertices, faces)

    # Check all components exist
    assert 'total' in loss_dict
    assert 'silhouette' in loss_dict
    assert 'edge' in loss_dict
    assert 'laplacian' in loss_dict

    # Total should be sum of weighted components
    assert total_loss.item() > 0

    print("✓ test_combined_loss passed")
    print("  Loss components:")
    for key, value in loss_dict.items():
        print(f"    {key}: {value:.6f}")


if __name__ == "__main__":
    print("Testing loss functions...")
    print()

    test_silhouette_loss()
    print()

    test_edge_loss()
    print()

    test_laplacian_loss()
    print()

    test_combined_loss()
    print()

    print("=" * 60)
    print("All loss function tests passed! ✓")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  • Silhouette loss: Measures shape matching")
    print("  • Edge loss: Keeps edge lengths uniform")
    print("  • Laplacian loss: Enforces smooth surfaces")
    print("  • Normal loss: Encourages consistent face orientations")
    print("  • Combined loss: Balances data fitting + regularization")
    print()
    print("Without regularization:")
    print("  ✗ Mesh can become jagged, spiky, or degenerate")
    print("  ✗ Triangles can collapse or stretch excessively")
    print()
    print("With regularization:")
    print("  ✓ Mesh stays smooth and well-formed")
    print("  ✓ Optimization is more stable")
    print("  ✓ Results look better!")
