import torch
import numpy as np
from typing import Tuple
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.structures import Meshes


def create_icosphere(subdivisions: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create an icosphere (sphere made of triangles) using PyTorch3D subdivision.

    Args:
        subdivisions: How many times to subdivide (more = smoother sphere)
                     0: 12 verts, 20 faces
                     1: 42 verts, 80 faces
                     2: 162 verts, 320 faces
                     3: 642 verts, 1280 faces
                     4: 2562 verts, 5120 faces
                     5: 10242 verts, 20480 faces

    Returns:
        vertices: (V, 3) tensor of 3D positions
        faces: (F, 3) tensor of triangle indices

    Example:
        vertices, faces = create_icosphere(subdivisions=1)
        print(f"Vertices shape: {vertices.shape}")  # (42, 3) for sub=1
    """
    # Start with icosahedron vertices (12 vertices, 20 faces)
    t = (1.0 + np.sqrt(5.0)) / 2.0  # Golden ratio

    vertices = np.array([
        [-1,  t,  0], [1,  t,  0], [-1, -t,  0], [1, -t,  0],
        [0, -1,  t], [0,  1,  t], [0, -1, -t], [0,  1, -t],
        [t,  0, -1], [t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ], dtype=np.float32)

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)

    # Convert to PyTorch tensors
    vertices = torch.tensor(vertices, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.int64)

    # Use PyTorch3D's SubdivideMeshes for proper subdivision
    if subdivisions > 0:
        # Create a Meshes object (batch size 1)
        mesh = Meshes(verts=[vertices], faces=[faces])

        # Subdivide
        subdivider = SubdivideMeshes()
        for _ in range(subdivisions):
            mesh = subdivider(mesh)

        # Extract vertices and faces
        vertices = mesh.verts_packed()
        faces = mesh.faces_packed()

    # Normalize vertices to unit sphere
    vertices = vertices / torch.norm(vertices, dim=1, keepdim=True)

    return vertices, faces


def create_deformed_sphere(deformation_type: str = 'ellipsoid',
                           params: dict = None,
                           subdivisions: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a deformed sphere as ground truth for test cases.

    Args:
        deformation_type: 'ellipsoid', 'bump', 'twist', or 'cube'
        params: Parameters for the deformation
        subdivisions: Subdivision level for the sphere (default: 2)

    Returns:
        vertices, faces tensors
    """
    vertices, faces = create_icosphere(subdivisions=subdivisions)

    if params is None:
        params = {}

    if deformation_type == 'ellipsoid':
        # Stretch along axes
        scale = params.get('scale', [1.5, 1.0, 0.7])  # x, y, z stretching
        vertices[:, 0] *= scale[0]
        vertices[:, 1] *= scale[1]
        vertices[:, 2] *= scale[2]

    elif deformation_type == 'bump':
        # Add a bump on one side
        amplitude = params.get('amplitude', 0.3)
        # Add bump where x > 0
        mask = vertices[:, 0] > 0
        vertices[mask, 0] += amplitude

    elif deformation_type == 'twist':
        # Twist around y-axis - rotation angle varies with y-coordinate
        angle = params.get('angle', np.pi / 4)

        # For each vertex, rotate around y-axis by an angle proportional to y
        for i in range(len(vertices)):
            y = vertices[i, 1].item()
            # Twist angle increases with height
            twist_angle = angle * y

            # Rotation matrix around y-axis
            cos_a = np.cos(twist_angle)
            sin_a = np.sin(twist_angle)

            x_old = vertices[i, 0].item()
            z_old = vertices[i, 2].item()

            # Apply rotation
            vertices[i, 0] = cos_a * x_old - sin_a * z_old
            vertices[i, 2] = sin_a * x_old + cos_a * z_old

    elif deformation_type == 'cube':
        # Project sphere onto cube surface - very challenging!
        # This creates sharp edges and flat faces from a smooth sphere
        size = params.get('size', 1.0)

        # For each vertex, find the dominant axis and push to cube face
        # This is equivalent to normalizing by L-infinity norm
        abs_coords = torch.abs(vertices)
        max_coord, _ = abs_coords.max(dim=1, keepdim=True)

        # Scale to cube by normalizing to max coordinate
        vertices = vertices / max_coord * size

    return vertices, faces


def save_mesh(vertices: torch.Tensor, faces: torch.Tensor, path: str):
    """
    Save mesh as Wavefront OBJ file.

    OBJ format:
        v x y z        # vertex
        f v1 v2 v3     # face (1-indexed!)
    """
    with open(path, 'w') as f:
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Write faces (OBJ uses 1-indexing, not 0-indexing!)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print(f"Saved mesh to {path}")


def load_mesh(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load mesh from OBJ file.
    """
    vertices = []
    faces = []

    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                # Parse vertex: "v 1.0 2.0 3.0"
                parts = line.strip().split()
                vertices.append(
                    [float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                # Parse face: "f 1 2 3" (convert to 0-indexed)
                parts = line.strip().split()
                # Handle "f 1/1/1 2/2/2 3/3/3" format (ignore texture/normal indices)
                face = [int(p.split('/')[0]) - 1 for p in parts[1:4]]
                faces.append(face)

    vertices = torch.tensor(vertices, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.int64)

    return vertices, faces


# ==================== UNIT TESTS ====================

def test_create_icosphere():
    """Test icosphere creation"""
    vertices, faces = create_icosphere(subdivisions=0)

    # Should have 12 vertices for base icosahedron
    assert vertices.shape[0] == 12, f"Expected 12 vertices, got {vertices.shape[0]}"
    assert faces.shape[0] == 20, f"Expected 20 faces, got {faces.shape[0]}"

    # Vertices should be on unit sphere
    distances = torch.norm(vertices, dim=1)
    assert torch.allclose(distances, torch.ones_like(distances), atol=1e-5), \
        "Vertices should be on unit sphere"

    print("✓ test_create_icosphere passed")


def test_deformed_sphere():
    """Test deformation"""
    vertices, faces = create_deformed_sphere(
        'ellipsoid', {'scale': [2.0, 1.0, 1.0]})

    # Check x-coordinates are stretched
    max_x = vertices[:, 0].max()
    max_y = vertices[:, 1].max()
    assert max_x > max_y * 1.5, "X should be stretched more than Y"

    print("✓ test_deformed_sphere passed")


def test_save_load_mesh():
    """Test save/load round trip"""
    import tempfile
    import os

    vertices, faces = create_icosphere()

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
        temp_path = f.name

    try:
        save_mesh(vertices, faces, temp_path)
        loaded_v, loaded_f = load_mesh(temp_path)

        # Check if loaded mesh matches original
        assert torch.allclose(vertices, loaded_v,
                              atol=1e-5), "Vertices don't match"
        assert torch.equal(faces, loaded_f), "Faces don't match"

        print("✓ test_save_load_mesh passed")
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    # This only runs when you directly execute: python src/mesh.py
    # It's for quick manual testing during development
    print("Creating test meshes for visualization...")

    import os
    os.makedirs("tests/manual_inspection", exist_ok=True)

    v, f = create_icosphere()
    save_mesh(v, f, "tests/manual_inspection/sphere.obj")

    v_def, f_def = create_deformed_sphere('ellipsoid')
    save_mesh(v_def, f_def, "tests/manual_inspection/ellipsoid.obj")

    print("Check tests/manual_inspection/ folder!")
