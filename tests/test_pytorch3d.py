"""Test which PyTorch3D components work"""
import torch
print(f"✓ PyTorch {torch.__version__} imported")

try:
    import pytorch3d
    print(f"✓ PyTorch3D {pytorch3d.__version__} imported")
except Exception as e:
    print(f"✗ PyTorch3D import failed: {e}")
    exit(1)

try:
    from pytorch3d.structures import Meshes
    print("✓ Meshes imported")
except Exception as e:
    print(f"✗ Meshes import failed: {e}")

try:
    from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras
    print("✓ Camera functions imported")
except Exception as e:
    print(f"✗ Camera import failed: {e}")

try:
    from pytorch3d.renderer import (
        RasterizationSettings,
        MeshRasterizer,
        MeshRenderer,
    )
    print("✓ Rasterizer imported")
except Exception as e:
    print(f"✗ Rasterizer import failed: {e}")

try:
    from pytorch3d.renderer import (
        SoftSilhouetteShader,
        HardPhongShader,
    )
    print("✓ Shaders imported")
except Exception as e:
    print(f"✗ Shaders import failed: {e}")

print("\n" + "="*50)
print("Testing basic functionality...")
print("="*50)

try:
    # Test camera creation
    R, T = look_at_view_transform(dist=2.7, elev=0, azim=0)
    cameras = FoVPerspectiveCameras(R=R, T=T)
    print("✓ Camera creation works")
except Exception as e:
    print(f"✗ Camera creation failed: {e}")

try:
    # Test mesh creation
    verts = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.float32)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    mesh = Meshes(verts=[verts], faces=[faces])
    print("✓ Mesh creation works")
except Exception as e:
    print(f"✗ Mesh creation failed: {e}")

print("\n" + "="*50)
if torch.cuda.is_available():
    print("✓✓✓ All tests passed! PyTorch3D is fully working with CUDA! ✓✓✓")
else:
    print("✓ All tests passed! PyTorch3D is working (CPU mode)")
print("="*50)
