#!/usr/bin/env python3
"""
Simple xformers build wrapper without Unicode characters
"""

import os
import sys
import subprocess
from pathlib import Path

def build_xformers():
    """Build xformers using the GitHub repository with proper CUDA 12.8 setup"""

    # Set up environment
    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

    env_vars = {
        "CUDA_PATH": cuda_path,
        "CUDA_HOME": cuda_path,
        "CUDA_TOOLKIT_ROOT_DIR": cuda_path,
        "FORCE_CUDA": "1",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",
        "XFORMERS_FORCE_CUDA_VERSION": "12.8"
    }

    # Update PATH
    current_path = os.environ.get("PATH", "")
    cuda_bin_path = os.path.join(cuda_path, "bin")
    new_path = f"{cuda_bin_path};{current_path}"

    print("Setting up CUDA 12.8 environment...")
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"  {var} = {value}")

    os.environ["PATH"] = new_path

    # Get the directory containing this script
    script_dir = Path(__file__).parent
    xformers_dir = script_dir / "wheels" / "xformers-source"

    if not xformers_dir.exists():
        print(f"ERROR: xformers source not found at {xformers_dir}")
        return False

    # Change to xformers directory
    original_cwd = os.getcwd()
    os.chdir(xformers_dir)

    try:
        # Apply PyTorch patch first
        print("Applying PyTorch CUDA version override patch...")

        import torch
        torch_path = Path(torch.__file__).parent
        cpp_extension_path = torch_path / "utils" / "cpp_extension.py"

        if cpp_extension_path.exists():
            with open(cpp_extension_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if "XFORMERS_FORCE_CUDA_VERSION" not in content:
                target_line = "cuda_str_version = cuda_version.group(1)"
                if target_line in content:
                    patch_code = """
    # Allow override via environment variable for xformers building
    override_cuda = os.environ.get('XFORMERS_FORCE_CUDA_VERSION')
    if override_cuda:
        cuda_str_version = override_cuda
"""
                    content = content.replace(target_line, target_line + patch_code)

                    with open(cpp_extension_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print("PyTorch patch applied successfully")

        # Build command
        build_cmd = [
            sys.executable, "setup.py", "bdist_wheel",
            "--dist-dir", "./wheels"
        ]

        print(f"Building xformers...")
        result = subprocess.run(
            build_cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )

        if result.returncode == 0:
            print("xformers build completed successfully")

            # Find the built wheel
            wheels_dir = xformers_dir / "wheels"
            wheel_files = list(wheels_dir.glob("*.whl"))

            if wheel_files:
                wheel_file = wheel_files[0]
                print(f"Built wheel: {wheel_file.name}")

                # Copy to main wheels directory
                main_wheels_dir = script_dir / "wheels"
                main_wheels_dir.mkdir(exist_ok=True)
                target_path = main_wheels_dir / wheel_file.name
                import shutil
                shutil.copy2(wheel_file, target_path)
                print(f"Copied wheel to {target_path}")

                return True
            else:
                print("ERROR: No wheel file found after build")
                return False
        else:
            print(f"Build failed with return code {result.returncode}")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("Build timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"Build error: {e}")
        return False
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    try:
        success = build_xformers()
        if success:
            print("xformers build completed successfully!")
        else:
            print("xformers build failed!")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nBuild cancelled by user")
        sys.exit(1)