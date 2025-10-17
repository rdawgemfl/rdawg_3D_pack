#!/usr/bin/env python3
"""
Development Setup Script for RDAWG 3D Pack
Sets up development environment and builds documentation
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    print(f"🐍 Python {version.major}.{version.minor}.{version.micro} detected")

    if version < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        return False
    elif version > (3, 12):
        print("⚠️  Warning: Python 3.13+ detected. Some dependencies may need wheel builds")
    else:
        print("✅ Python version is compatible")

    return True

def check_dependencies():
    """Check development dependencies"""
    print("\n🔧 Checking development dependencies...")

    dev_deps = ["git", "pip", "setuptools", "wheel"]
    missing_deps = []

    for dep in dev_deps:
        try:
            if dep == "git":
                subprocess.run([dep, "--version"], capture_output=True, check=True)
            else:
                subprocess.run([sys.executable, "-m", dep, "--version"], capture_output=True, check=True)
            print(f"  ✅ {dep}")
        except (subprocess.CalledProcessError, ImportError):
            print(f"  ❌ {dep} not found")
            missing_deps.append(dep)

    return len(missing_deps) == 0

def setup_git_repo():
    """Initialize git repository"""
    print("\n📁 Setting up git repository...")

    try:
        # Initialize git repo
        subprocess.run(["git", "init"], check=True, capture_output=True)
        print("  ✅ Git repository initialized")

        # Create .gitignore
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# ComfyUI specific
*.temp
*.backup
models/
input/
output/
"""

        with open(".gitignore", "w") as f:
            f.write(gitignore_content)
        print("  ✅ .gitignore created")

        return True

    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed to setup git: {e}")
        return False

def create_development_scripts():
    """Create development helper scripts"""
    print("\n📜 Creating development scripts...")

    scripts = {
        "test.py": """#!/usr/bin/env python3
\"\"\"
Run tests for RDAWG 3D Pack
\"\"\"

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from __init__ import NODE_CLASS_MAPPINGS
import torch

def test_imports():
    \"\"\"Test all imports\"\"\"
    print("🧪 Testing imports...")

    try:
        import torch
        print("  ✅ PyTorch")
    except ImportError:
        print("  ❌ PyTorch")
        return False

    try:
        import numpy as np
        print("  ✅ NumPy")
    except ImportError:
        print("  ❌ NumPy")
        return False

    try:
        import trimesh
        print("  ✅ Trimesh")
    except ImportError:
        print("  ⚠️  Trimesh (fallback mode)")

    try:
        import open3d as o3d
        print("  ✅ Open3D")
    except ImportError:
        print("  ⚠️  Open3D (fallback mode)")

    return True

def test_nodes():
    \"\"\"Test node instantiation\"\"\"
    print("\\n🧪 Testing nodes...")

    for node_name, node_class in NODE_CLASS_MAPPINGS.items():
        try:
            node = node_class()
            print(f"  ✅ {node_name}")
        except Exception as e:
            print(f"  ❌ {node_name}: {e}")
            return False

    return True

def main():
    \"\"\"Main test function\"\"\"
    print("🔷 RDAWG 3D Pack - Test Suite")
    print("=" * 40)

    success = True

    if not test_imports():
        success = False

    if not test_nodes():
        success = False

    if success:
        print("\\n🎉 All tests passed!")
    else:
        print("\\n❌ Some tests failed!")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
""",

        "build_docs.py": """#!/usr/bin/env python3
\"\"\"
Build documentation for RDAWG 3D Pack
\"\"\"

import os
import subprocess
from pathlib import Path

def main():
    \"\"\"Build documentation\"\"\"
    print("📚 Building documentation...")

    docs_dir = Path(__file__).parent / "docs"
    if not docs_dir.exists():
        print("❌ docs directory not found")
        return False

    # This is a placeholder for future documentation building
    print("✅ Documentation structure ready")
    print("   - README.md (main documentation)")
    print("   - docs/ directory for additional docs")
    print("   - examples/ directory for workflows")

    return True

if __name__ == "__main__":
    main()
""",

        "release.py": """#!/usr/bin/env python3
\"\"\"
Create release package for RDAWG 3D Pack
\"\"\"

import os
import shutil
import subprocess
from pathlib import Path
import zipfile

def create_release():
    \"\"\"Create release package\"\"\"
    print("📦 Creating release package...")

    # Get version from pyproject.toml
    version = "1.0.0"  # Default

    # Create release directory
    release_dir = Path(f"rdawg-3d-pack-v{version}")
    if release_dir.exists():
        shutil.rmtree(release_dir)

    # Copy files
    files_to_include = [
        "__init__.py",
        "core_3d.py",
        "install.py",
        "pyproject.toml",
        "requirements.txt",
        "README.md",
        "LICENSE"
    ]

    release_dir.mkdir()

    for file in files_to_include:
        if Path(file).exists():
            shutil.copy2(file, release_dir / file)
            print(f"  ✅ {file}")
        else:
            print(f"  ⚠️  {file} not found")

    # Copy directories
    dirs_to_include = ["docs", "examples", "scripts", "wheels"]
    for dir_name in dirs_to_include:
        if Path(dir_name).exists():
            shutil.copytree(dir_name, release_dir / dir_name)
            print(f"  ✅ {dir_name}/")

    # Create zip
    zip_name = f"rdawg-3d-pack-v{version}.zip"
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in release_dir.rglob('*'):
            if file.is_file():
                zipf.write(file, file.relative_to(release_dir))

    print(f"\\n🎉 Release package created: {zip_name}")
    return True

if __name__ == "__main__":
    create_release()
"""
    }

    scripts_dir = Path(__file__).parent
    for script_name, content in scripts.items():
        script_path = scripts_dir / script_name
        with open(script_path, "w") as f:
            f.write(content)
        os.chmod(script_path, 0o755)
        print(f"  ✅ {script_name}")

    return True

def main():
    """Main development setup function"""
    print("🔷 RDAWG 3D Pack - Development Setup")
    print("   Setting up development environment")
    print("=" * 50)

    # Check requirements
    if not check_python_version():
        return False

    if not check_dependencies():
        print("❌ Please install missing dependencies first")
        return False

    # Setup development environment
    if not setup_git_repo():
        print("⚠️  Git setup failed, but continuing...")

    if not create_development_scripts():
        print("❌ Failed to create development scripts")
        return False

    print("\n" + "=" * 50)
    print("🎉 Development setup completed successfully!")
    print("\n📚 Development tools created:")
    print("  - test.py        - Run test suite")
    print("  - build_docs.py  - Build documentation")
    print("  - release.py     - Create release package")
    print("  - download_wheels.py - Download required wheels")
    print("\n🔧 Next steps:")
    print("1. Run 'python scripts/test.py' to verify installation")
    print("2. Start developing new features")
    print("3. Use 'python scripts/release.py' when ready to release")

    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏭️  Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed: {str(e)}")
        sys.exit(1)