#!/usr/bin/env python3
"""
Simple test script for RDAWG 3D Pack
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test all imports"""
    print("Testing imports...")

    try:
        import torch
        print(f"  [OK] PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"  [FAIL] PyTorch: {e}")
        return False

    try:
        import numpy as np
        print(f"  [OK] NumPy {np.__version__}")
    except ImportError as e:
        print(f"  [FAIL] NumPy: {e}")
        return False

    try:
        import trimesh
        print(f"  [OK] Trimesh {trimesh.__version__}")
    except ImportError as e:
        print(f"  [WARN] Trimesh not available: {e}")

    try:
        import open3d as o3d
        print(f"  [OK] Open3D {o3d.__version__}")
    except ImportError as e:
        print(f"  [WARN] Open3D not available: {e}")

    return True

def test_nodes():
    """Test node instantiation"""
    print("\nTesting nodes...")

    try:
        from __init__ import NODE_CLASS_MAPPINGS

        for node_name, node_class in NODE_CLASS_MAPPINGS.items():
            try:
                node = node_class()
                print(f"  [OK] {node_name}")
            except Exception as e:
                print(f"  [FAIL] {node_name}: {e}")
                return False

        print(f"  [OK] Total nodes loaded: {len(NODE_CLASS_MAPPINGS)}")
        return True

    except ImportError as e:
        print(f"  [FAIL] Could not import nodes: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")

    try:
        import torch
        import numpy as np
        from __init__ import NODE_CLASS_MAPPINGS

        # Test CreateMesh node
        create_mesh = NODE_CLASS_MAPPINGS["RDAWG3DCreateMesh"]()

        # Create simple cube
        vertices = torch.tensor([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ], dtype=torch.float32)

        faces = torch.tensor([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 7, 6], [4, 6, 5],  # top
            [0, 4, 5], [0, 5, 1],  # front
            [2, 6, 7], [2, 7, 3],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2]   # right
        ], dtype=torch.long)

        mesh, info = create_mesh.create_mesh(vertices, faces, "auto")
        print(f"  [OK] Created cube mesh: {len(mesh['vertices'])} vertices, {len(mesh['faces'])} faces")

        # Test Transform node
        transform = NODE_CLASS_MAPPINGS["RDAWG3DTransform"]()
        transformed_mesh = transform.transform_mesh(mesh, 2.0, 45.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        print(f"  [OK] Transformed mesh successfully")

        # Test MeshToImage node
        mesh_to_image = NODE_CLASS_MAPPINGS["RDAWG3DMeshToImage"]()
        image = mesh_to_image.render_mesh_to_image(
            transformed_mesh[0], 256, 256, "#FFFFFF", "#000000", True, 2.0, 0.0, 0.0
        )
        print(f"  [OK] Rendered image: {image[0].shape}")

        return True

    except Exception as e:
        print(f"  [FAIL] Basic functionality test: {e}")
        return False

def main():
    """Main test function"""
    print("RDAWG 3D Pack - Test Suite")
    print("=" * 40)

    success = True

    if not test_imports():
        success = False

    if not test_nodes():
        success = False

    if not test_basic_functionality():
        success = False

    print("\n" + "=" * 40)
    if success:
        print("SUCCESS: All tests passed!")
        print("RDAWG 3D Pack is ready for use!")
    else:
        print("FAILURE: Some tests failed!")
        print("Please check the errors above.")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)