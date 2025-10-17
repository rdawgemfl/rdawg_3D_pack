#!/usr/bin/env python3
"""
Download Required Wheels for RDAWG 3D Pack
Automated wheel downloader for optimal dependencies
"""

import os
import sys
import urllib.request
import subprocess
from pathlib import Path

# Wheel download configurations
WHEELS_CONFIG = {
    "open3d": {
        "url": "https://github.com/isl-org/Open3D/releases/download/v0.19.0/open3d-0.19.0-cp311-cp311-win_amd64.whl",
        "filename": "open3d-0.19.0-cp311-cp311-win_amd64.whl",
        "description": "Open3D 0.19.0 for Python 3.11 Windows x64"
    },
    "trimesh": {
        "url": "https://files.pythonhosted.org/packages/d3/f1/3eb44efb7f4e4341d8646b7f684b4c0fbb26f7d7b5b2d2f7d1d8b0c5c7/trimesh-4.0.5-py3-none-any.whl",
        "filename": "trimesh-4.0.5-py3-none-any.whl",
        "description": "Trimesh 4.0.5 - Core mesh processing library"
    }
}

def download_wheel(wheel_name, config):
    """Download a single wheel file"""
    url = config["url"]
    filename = config["filename"]
    description = config["description"]

    print(f"\n🔧 Downloading {description}")
    print(f"   From: {url}")
    print(f"   To: {filename}")

    try:
        # Download with progress
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0

            with open(filename, 'wb') as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Show progress
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"   Progress: {progress:.1f}% ({downloaded}/{total_size} bytes)", end='\r')

            print(f"\n   ✅ {filename} downloaded successfully")
            return True

    except Exception as e:
        print(f"\n   ❌ Failed to download {filename}: {str(e)}")
        return False

def main():
    """Main download function"""
    print("🔷 RDAWG 3D Pack - Wheel Downloader")
    print("   Downloading required wheels for optimal performance")
    print("=" * 60)

    # Create wheels directory
    wheels_dir = Path(__file__).parent / "wheels"
    wheels_dir.mkdir(exist_ok=True)

    # Change to wheels directory
    os.chdir(wheels_dir)

    success_count = 0
    total_count = len(WHEELS_CONFIG)

    # Download each wheel
    for wheel_name, config in WHEELS_CONFIG.items():
        if download_wheel(wheel_name, config):
            success_count += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Download Summary: {success_count}/{total_count} wheels downloaded successfully")

    if success_count == total_count:
        print("🎉 All required wheels are ready!")
        print("\n📦 Next steps:")
        print("1. Install RDAWG 3D Pack in ComfyUI")
        print("2. Use install.py to install dependencies")
        print("3. Wheels will be used automatically for optimal performance")
    else:
        print("⚠️  Some wheels failed to download. You can install them manually or use pip install.")

    return success_count == total_count

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏭️  Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Download failed: {str(e)}")
        sys.exit(1)