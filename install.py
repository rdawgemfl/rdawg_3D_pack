#!/usr/bin/env python3
"""
RDAWG 3D Pack Installation Script
CUDA 12.8 + PyTorch 2.9.0 Optimized Version
"""

import sys
import subprocess
import importlib.util
import os

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_pytorch_cuda():
    """Install PyTorch with CUDA 12.8 support"""
    print("🔧 Installing PyTorch 2.9.0 with CUDA 12.8 support...")

    # Install PyTorch CUDA version from PyTorch index
    pytorch_cmd = [
        sys.executable, "-m", "pip", "install",
        "torch==2.9.0+cu128",
        "torchvision==0.24.0+cu128",
        "torchaudio==2.9.0+cu128",
        "--index-url", "https://download.pytorch.org/whl/cu128"
    ]

    try:
        print("  Downloading PyTorch CUDA wheels (this may take a few minutes)...")
        result = subprocess.run(pytorch_cmd, capture_output=True, text=True, timeout=600)

        if result.returncode == 0:
            print("  ✅ PyTorch 2.9.0+cu128 installed successfully")
            return True
        else:
            print(f"  ❌ Failed to install PyTorch CUDA: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("  ⏰ PyTorch installation timed out")
        return False
    except Exception as e:
        print(f"  ❌ Error installing PyTorch: {str(e)}")
        return False

def check_pytorch_version():
    """Check if PyTorch version is compatible"""
    try:
        import torch
        version = torch.__version__
        print(f"✅ PyTorch {version} detected")

        # Check if CUDA is available
        if torch.cuda.is_available():
            print(f"✅ CUDA {torch.version.cuda} available")
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠️  Warning: CUDA not available, installing CUDA version...")
            return install_pytorch_cuda()

    except ImportError:
        print("❌ PyTorch not found, installing CUDA version...")
        return install_pytorch_cuda()

def install_core_dependencies():
    """Install core dependencies"""
    print("🔧 Installing core dependencies...")

    core_packages = [
        "trimesh>=4.0.0",
        "tqdm>=4.65.0",
        "einops>=0.7.0",
    ]

    for package in core_packages:
        try:
            print(f"  Installing {package}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print(f"  ✅ {package} installed successfully")
            else:
                print(f"  ❌ Failed to install {package}: {result.stderr}")
                return False

        except Exception as e:
            print(f"  ❌ Error installing {package}: {str(e)}")
            return False

    return True

def install_optional_dependencies():
    """Install optional dependencies based on user choice"""
    print("\n📦 Optional dependencies available:")
    print("1. Full 3D processing suite (open3d, pytorch3d, etc.)")
    print("2. Neural rendering capabilities")
    print("3. 3D reconstruction tools")
    print("4. Skip optional dependencies")

    try:
        choice = input("\nSelect option (1-4): ").strip()

        if choice == "1":
            return install_full_suite()
        elif choice == "2":
            return install_neural_rendering()
        elif choice == "3":
            return install_reconstruction()
        elif choice == "4":
            print("⏭️  Skipping optional dependencies")
            return True
        else:
            print("❌ Invalid choice")
            return False

    except KeyboardInterrupt:
        print("\n⏭️  Installation cancelled")
        return True

def install_full_suite():
    """Install full 3D processing suite"""
    print("🔧 Installing full 3D processing suite...")

    packages = [
        "open3d>=0.18.0",
        "pytorch3d>=0.7.5",
        "scipy>=1.11.0",
        "scikit-image>=0.21.0",
        "matplotlib>=3.7.0",
        "plotly>=5.15.0",
        "assimp>=5.3.0",
        "gltflib>=1.16.0",
        "plyfile>=1.0.0",
    ]

    return install_packages(packages)

def install_neural_rendering():
    """Install neural rendering dependencies"""
    print("🔧 Installing neural rendering dependencies...")

    packages = [
        "transformers>=4.35.0",
        "diffusers>=0.24.0",
        "accelerate>=0.24.0",
        "xformers>=0.0.32",
    ]

    return install_packages(packages)

def install_reconstruction():
    """Install 3D reconstruction dependencies"""
    print("🔧 Installing 3D reconstruction dependencies...")

    packages = [
        "nerfacc>=0.5.0",
    ]

    return install_packages(packages)

def install_packages(packages):
    """Install a list of packages"""
    failed_packages = []

    for package in packages:
        try:
            print(f"  Installing {package}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print(f"  ✅ {package} installed successfully")
            else:
                print(f"  ❌ Failed to install {package}")
                print(f"     Error: {result.stderr}")
                failed_packages.append(package)

        except subprocess.TimeoutExpired:
            print(f"  ⏰ Timeout installing {package}")
            failed_packages.append(package)
        except Exception as e:
            print(f"  ❌ Error installing {package}: {str(e)}")
            failed_packages.append(package)

    if failed_packages:
        print(f"\n⚠️  Failed to install: {', '.join(failed_packages)}")
        print("   You can install them manually later if needed")
        return False

    return True

def create_example_workflow():
    """Create example workflow file"""
    example_workflow = {
        "last_node_id": 5,
        "last_link_id": 4,
        "nodes": {
            "1": {
                "inputs": {
                    "file_path": "path/to/your/model.obj",
                    "load_texture": True,
                    "normalize": True,
                    "device": "auto"
                },
                "class_type": "RDAWG3DLoadModel",
                "pos": [50, 50]
            },
            "2": {
                "inputs": {
                    "mesh": ["1", 0],
                    "scale": 1.0,
                    "rotation_x": 0.0,
                    "rotation_y": 45.0,
                    "rotation_z": 0.0,
                    "translate_x": 0.0,
                    "translate_y": 0.0,
                    "translate_z": 0.0
                },
                "class_type": "RDAWG3DTransform",
                "pos": [300, 50]
            },
            "3": {
                "inputs": {
                    "mesh": ["2", 0],
                    "width": 512,
                    "height": 512,
                    "background_color": "#000000",
                    "mesh_color": "#FFFFFF",
                    "light_intensity": 1.0,
                    "camera_distance": 2.0,
                    "elevation": 0.0,
                    "azimuth": 0.0
                },
                "class_type": "RDAWG3DMeshToImage",
                "pos": [550, 50]
            },
            "4": {
                "inputs": {
                    "images": ["3", 0],
                    "filename_prefix": "RDAWG_3D_Render",
                },
                "class_type": "SaveImage",
                "pos": [800, 50]
            }
        },
        "links": [
            [1, 0, 2, 0],
            [2, 0, 3, 0],
            [3, 0, 4, 0]
        ],
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4
    }

    try:
        import json
        with open("example_workflow.json", "w") as f:
            json.dump(example_workflow, f, indent=2)
        print("✅ Example workflow created: example_workflow.json")
    except Exception as e:
        print(f"⚠️  Could not create example workflow: {str(e)}")

def main():
    """Main installation function"""
    print("🔷 RDAWG 3D Pack Installation")
    print("   CUDA 12.8 + PyTorch 2.9.0 Optimized")
    print("=" * 50)

    # Check system requirements
    if not check_python_version():
        return False

    if not check_pytorch_version():
        return False

    # Install core dependencies
    if not install_core_dependencies():
        print("❌ Core dependency installation failed")
        return False

    # Install optional dependencies
    if not install_optional_dependencies():
        print("⚠️  Optional dependency installation had issues")

    # Create example workflow
    create_example_workflow()

    print("\n" + "=" * 50)
    print("🎉 Installation completed successfully!")
    print("🔷 RDAWG 3D Pack is ready to use!")
    print("\n📚 Next steps:")
    print("1. Restart ComfyUI")
    print("2. Look for 'RDAWG 3D' nodes in the node menu")
    print("3. Load example_workflow.json for a quick start")
    print("\n📖 Documentation: README.md")
    print("🐛 Issues: https://github.com/rdawgemfl/rdawg-3d-pack/issues")

    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏭️  Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Installation failed: {str(e)}")
        sys.exit(1)