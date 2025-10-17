#!/usr/bin/env python3
"""
Open3D Wheel Downloader for RDAWG 3D Pack
Automatically downloads the correct Open3D wheel for your Python version
"""

import os
import sys
import platform
import subprocess
from urllib.request import urlretrieve

def get_python_version():
    """Get Python version in format cp311, cp312, etc."""
    version = sys.version_info
    return f"cp{version.major}{version.minor}"

def get_platform_tag():
    """Get platform tag for wheel download"""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows":
        return "win_amd64"
    elif system == "linux":
        if machine in ["x86_64", "amd64"]:
            return "linux_x86_64"
        else:
            return f"linux_{machine}"
    elif system == "darwin":
        if machine == "arm64" or platform.processor() == "arm":
            return "macosx_11_0_arm64"
        else:
            return "macosx_10_9_x86_64"
    else:
        raise ValueError(f"Unsupported platform: {system}")

def download_open3d_wheel():
    """Download the correct Open3D wheel for current environment"""
    python_version = get_python_version()
    platform_tag = get_platform_tag()

    wheel_name = f"open3d-0.19.0-{python_version}-{python_version}-{platform_tag}.whl"
    download_url = f"https://github.com/isl-org/Open3D/releases/download/v0.19.0/{wheel_name}"

    print(f"🔷 RDAWG 3D Pack - Open3D Wheel Downloader")
    print(f"Python Version: {python_version}")
    print(f"Platform: {platform_tag}")
    print(f"Wheel: {wheel_name}")
    print(f"Download URL: {download_url}")
    print()

    try:
        print("📥 Downloading Open3D wheel...")
        urlretrieve(download_url, wheel_name)
        print(f"✅ Successfully downloaded: {wheel_name}")

        print(f"📦 To install, run:")
        print(f"   pip install {wheel_name}")

        return wheel_name

    except Exception as e:
        print(f"❌ Download failed: {e}")
        print()
        print("🌐 Please download manually from:")
        print("   https://github.com/isl-org/Open3D/releases/tag/v0.19.0")
        return None

def main():
    """Main function"""
    print("Open3D 0.19.0 Wheel Downloader for RDAWG 3D Pack")
    print("=" * 50)

    # Check if Open3D is already installed
    try:
        import open3d
        print(f"✅ Open3D {open3d.__version__} is already installed!")
        return
    except ImportError:
        print("📦 Open3D not found, downloading wheel...")

    # Download wheel
    wheel_file = download_open3d_wheel()

    if wheel_file:
        # Ask if user wants to install
        response = input("\nInstall Open3D now? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", wheel_file], check=True)
                print("✅ Open3D installed successfully!")

                # Clean up downloaded wheel
                os.remove(wheel_file)
                print(f"🧹 Cleaned up {wheel_file}")

            except subprocess.CalledProcessError as e:
                print(f"❌ Installation failed: {e}")
                print(f"💡 Please install manually: pip install {wheel_file}")

if __name__ == "__main__":
    main()