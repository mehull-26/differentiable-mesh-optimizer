"""
Demo script to visualize how different losses affect optimization.

This script demonstrates:
1. What happens WITHOUT regularization (data loss only)
2. What happens WITH regularization (data + regularization)
3. The importance of balancing loss weights
"""

from src.losses import (
    silhouette_loss,
    edge_loss,
    laplacian_loss,
    combined_loss
)
from src.mesh import create_icosphere
import numpy as np
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


print("=" * 70)
print("LOSS FUNCTION DEMONSTRATION")
print("=" * 70)
print()

# Create a simple mesh
vertices, faces = create_icosphere(subdivisions=1)
print(f"Mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
print()

# ============================================================
# Demo 1: Edge Loss Behavior
# ============================================================
print("Demo 1: Edge Loss - Preventing Deformation")
print("-" * 70)

initial_edge_loss = edge_loss(vertices, faces)
print(f"Initial edge loss (uniform sphere): {initial_edge_loss.item():.6f}")

# Stretch the mesh
vertices_stretched = vertices.clone()
vertices_stretched[:, 0] *= 3.0  # Extreme stretch along X
stretched_edge_loss = edge_loss(vertices_stretched, faces)
print(f"Edge loss after 3x stretch:         {stretched_edge_loss.item():.6f}")
print(
    f"Increase: {stretched_edge_loss.item() / (initial_edge_loss.item() + 1e-8):.2f}x")
print()
print("👉 Edge loss PENALIZES non-uniform deformations")
print("   This keeps the mesh from becoming too distorted")
print()

# ============================================================
# Demo 2: Laplacian Loss Behavior
# ============================================================
print("Demo 2: Laplacian Loss - Enforcing Smoothness")
print("-" * 70)

smooth_lap_loss = laplacian_loss(vertices, faces)
print(f"Laplacian loss (smooth sphere):  {smooth_lap_loss.item():.6f}")

# Add random noise (make it rough/spiky)
vertices_noisy = vertices + torch.randn_like(vertices) * 0.2
noisy_lap_loss = laplacian_loss(vertices_noisy, faces)
print(f"Laplacian loss (noisy sphere):   {noisy_lap_loss.item():.6f}")
print(f"Increase: {noisy_lap_loss.item() / smooth_lap_loss.item():.2f}x")
print()
print("👉 Laplacian loss PENALIZES rough, spiky surfaces")
print("   This encourages smooth, natural-looking meshes")
print()

# ============================================================
# Demo 3: Combined Loss Balance
# ============================================================
print("Demo 3: Combined Loss - Balancing Act")
print("-" * 70)

# Simulate target and rendered silhouettes
target = torch.ones(1, 64, 64, 1)
rendered_good = torch.ones(1, 64, 64, 1) * 0.95  # Close match
rendered_bad = torch.ones(1, 64, 64, 1) * 0.5    # Poor match

# Scenario 1: Good data match, smooth mesh
loss1, dict1 = combined_loss(rendered_good, target, vertices, faces)
print("Scenario 1: Good match + smooth mesh")
print(f"  Silhouette loss: {dict1['silhouette']:.6f} (low = good match)")
print(f"  Laplacian loss:  {dict1['laplacian']:.6f} (low = smooth)")
print(f"  Total loss:      {dict1['total']:.6f}")
print()

# Scenario 2: Poor data match, smooth mesh
loss2, dict2 = combined_loss(rendered_bad, target, vertices, faces)
print("Scenario 2: Poor match + smooth mesh")
print(f"  Silhouette loss: {dict2['silhouette']:.6f} (high = bad match)")
print(f"  Laplacian loss:  {dict2['laplacian']:.6f} (low = smooth)")
print(f"  Total loss:      {dict2['total']:.6f}")
print()

# Scenario 3: Good data match, noisy mesh
loss3, dict3 = combined_loss(rendered_good, target, vertices_noisy, faces)
print("Scenario 3: Good match + noisy mesh")
print(f"  Silhouette loss: {dict3['silhouette']:.6f} (low = good match)")
print(f"  Laplacian loss:  {dict3['laplacian']:.6f} (high = not smooth)")
print(f"  Total loss:      {dict3['total']:.6f}")
print()

print("👉 Combined loss balances:")
print("   • Data term: Match the target")
print("   • Regularization: Keep mesh well-behaved")
print()

# ============================================================
# Demo 4: Effect of Regularization Weights
# ============================================================
print("Demo 4: Weight Tuning")
print("-" * 70)

# Very high regularization weight
weights_high_reg = {'silhouette': 1.0,
                    'laplacian': 10.0, 'edge': 1.0, 'normal': 0.1}
loss_high_reg, _ = combined_loss(
    rendered_bad, target, vertices, faces, weights_high_reg)

# Very low regularization weight
weights_low_reg = {'silhouette': 1.0,
                   'laplacian': 0.001, 'edge': 0.001, 'normal': 0.0}
loss_low_reg, _ = combined_loss(
    rendered_bad, target, vertices, faces, weights_low_reg)

print("Same mesh with different weights:")
print(f"  High regularization weight: {loss_high_reg.item():.6f}")
print(f"  Low regularization weight:  {loss_low_reg.item():.6f}")
print()
print("Effect:")
print("  • High reg weight → Prioritizes smooth mesh")
print("                    → May not match target perfectly")
print("  • Low reg weight  → Prioritizes matching target")
print("                    → Mesh may become jagged")
print()

# ============================================================
# Summary
# ============================================================
print("=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print()
print("1. SILHOUETTE LOSS (Data Loss)")
print("   • Measures: How well shape matches target")
print("   • Goal: Minimize difference between rendered and target")
print("   • Range: 0 (perfect) to 1 (completely wrong)")
print()
print("2. EDGE LOSS (Regularization)")
print("   • Measures: Deviation from uniform edge lengths")
print("   • Prevents: Stretched or collapsed triangles")
print("   • Without it: Mesh becomes distorted")
print()
print("3. LAPLACIAN LOSS (Regularization)")
print("   • Measures: Surface roughness")
print("   • Prevents: Spiky, jagged surfaces")
print("   • Without it: Mesh looks unnatural")
print()
print("4. NORMAL CONSISTENCY (Regularization)")
print("   • Measures: Face orientation consistency")
print("   • Prevents: Sharp discontinuities")
print("   • Without it: Visual artifacts")
print()
print("5. WEIGHT BALANCING")
print("   • Too much data weight → Overfitting, ugly mesh")
print("   • Too much reg weight → Smooth but doesn't match")
print("   • Balance is key! Start with defaults and tune")
print()
print("OPTIMIZATION WITHOUT REGULARIZATION:")
print("  ❌ Mesh can invert (inside-out)")
print("  ❌ Triangles can collapse to points")
print("  ❌ Edges can stretch infinitely")
print("  ❌ Surface becomes spiky and chaotic")
print()
print("OPTIMIZATION WITH REGULARIZATION:")
print("  ✅ Mesh stays well-formed")
print("  ✅ Smooth, natural surfaces")
print("  ✅ Stable optimization")
print("  ✅ Visually pleasing results")
print()
