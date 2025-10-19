#!/usr/bin/env python3
"""
RDAWG 3D Pack - xformers Build Script
Builds xformers from source for PyTorch 2.9.0 + Python 3.12.10 + CUDA 12.8
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def setup_cuda_environment():
    """Set up CUDA 12.8 environment variables"""
    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

    env_vars = {
        "CUDA_PATH": cuda_path,
        "CUDA_HOME": cuda_path,
        "CUDA_TOOLKIT_ROOT_DIR": cuda_path,
        "FORCE_CUDA": "1",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",
        "XFORMERS_FORCE_CUDA_VERSION": "12.8"
    }

    # Update PATH to prioritize CUDA 12.8
    current_path = os.environ.get("PATH", "")
    cuda_bin_path = os.path.join(cuda_path, "bin")
    new_path = f"{cuda_bin_path};{current_path}"

    print("🔧 Setting up CUDA 12.8 environment...")
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"  {var} = {value}")

    os.environ["PATH"] = new_path
    print(f"  PATH = {new_path[:100]}...")

    return cuda_path

def patch_pytorch_cpp_extension():
    """Apply CUDA version override patch to PyTorch's cpp_extension.py"""
    print("🔧 Applying PyTorch CUDA version override patch...")

    try:
        # Find PyTorch installation
        import torch
        torch_path = Path(torch.__file__).parent
        cpp_extension_path = torch_path / "utils" / "cpp_extension.py"

        if not cpp_extension_path.exists():
            print(f"  ❌ cpp_extension.py not found at {cpp_extension_path}")
            return False

        # Read the file
        with open(cpp_extension_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if patch is already applied
        if "XFORMERS_FORCE_CUDA_VERSION" in content:
            print("  ✅ PyTorch patch already applied")
            return True

        # Find the CUDA version check function and apply patch
        patch_code = """
    # Allow override via environment variable for xformers building
    override_cuda = os.environ.get('XFORMERS_FORCE_CUDA_VERSION')
    if override_cuda:
        cuda_str_version = override_cuda
"""

        # Insert the patch after cuda_str_version is determined
        target_line = "cuda_str_version = cuda_version.group(1)"
        if target_line in content:
            content = content.replace(target_line, target_line + patch_code)

            # Also update the error handling to allow override
            old_error = "if cuda_ver.major != torch_cuda_version.major:\n            raise RuntimeError(CUDA_MISMATCH_MESSAGE, cuda_str_version, torch.version.cuda)"
            new_error = """if cuda_ver.major != torch_cuda_version.major:
            # Allow override via environment variable
            if not override_cuda:
                raise RuntimeError(CUDA_MISMATCH_MESSAGE, cuda_str_version, torch.version.cuda)
            else:
                logger.warning(f"Overriding CUDA version check: using {override_cuda} instead of {torch.version.cuda}")"""

            content = content.replace(old_error, new_error)

            # Write patched file
            with open(cpp_extension_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print("  ✅ PyTorch patch applied successfully")
            return True
        else:
            print("  ❌ Target line not found in cpp_extension.py")
            return False

    except Exception as e:
        print(f"  ❌ Failed to apply PyTorch patch: {e}")
        return False

def build_xformers():
    """Build xformers from source"""
    print("🔧 Building xformers from source...")

    # Get the directory containing this script
    script_dir = Path(__file__).parent
    xformers_dir = script_dir / "wheels" / "xformers-source"

    if not xformers_dir.exists():
        print(f"  ❌ xformers source not found at {xformers_dir}")
        return False

    # Change to xformers directory
    original_cwd = os.getcwd()
    os.chdir(xformers_dir)

    try:
        # Build command
        build_cmd = [
            sys.executable, "setup.py", "bdist_wheel",
            "--dist-dir", "./wheels"
        ]

        print(f"  Running: {' '.join(build_cmd)}")
        result = subprocess.run(
            build_cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )

        if result.returncode == 0:
            print("  ✅ xformers build completed successfully")

            # Find the built wheel
            wheels_dir = xformers_dir / "wheels"
            wheel_files = list(wheels_dir.glob("*.whl"))

            if wheel_files:
                wheel_file = wheel_files[0]
                print(f"  📦 Built wheel: {wheel_file.name}")

                # Copy to main wheels directory
                main_wheels_dir = script_dir / "wheels"
                main_wheels_dir.mkdir(exist_ok=True)
                shutil.copy2(wheel_file, main_wheels_dir / wheel_file.name)
                print(f"  ✅ Copied wheel to {main_wheels_dir}")

                return True
            else:
                print("  ❌ No wheel file found after build")
                return False
        else:
            print(f"  ❌ Build failed with return code {result.returncode}")
            print(f"  stdout: {result.stdout}")
            print(f"  stderr: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("  ⏰ Build timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"  ❌ Build error: {e}")
        return False
    finally:
        os.chdir(original_cwd)

def main():
    """Main build function"""
    print("🔷 RDAWG 3D Pack - xformers Build Script")
    print("   Building xformers for PyTorch 2.9.0 + Python 3.12.10 + CUDA 12.8")
    print("=" * 60)

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False

    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA {torch.version.cuda} available")
        else:
            print("⚠️  CUDA not available in PyTorch")
    except ImportError:
        print("❌ PyTorch not found")
        return False

    # Setup CUDA environment
    if not setup_cuda_environment():
        print("❌ Failed to setup CUDA environment")
        return False

    # Apply PyTorch patch
    if not patch_pytorch_cpp_extension():
        print("❌ Failed to apply PyTorch patch")
        return False

    # Build xformers
    if not build_xformers():
        print("❌ Failed to build xformers")
        return False

    print("\n" + "=" * 60)
    print("🎉 xformers build completed successfully!")
    print("📦 Built wheel is available in the wheels directory")
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏭️  Build cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        sys.exit(1)