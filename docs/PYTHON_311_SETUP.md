# Python 3.11 Setup Guide

## Why Python 3.11 is Recommended

Python 3.11 provides the **optimal balance** of compatibility and performance for the RDAWG 3D Pack:

### ✅ **Perfect Open3D Support**
- Open3D 0.19.0 has official pre-built wheels for Python 3.11
- No compilation required - instant installation
- Full feature support with all optimizations

### ✅ **PyTorch CUDA Integration**
- Excellent CUDA 12.8 support
- Best performance for GPU-accelerated 3D processing
- Stable memory management

### ✅ **3D Library Ecosystem**
- Trimesh 4.8.3 fully compatible
- All scientific computing libraries optimized
- Better stability than Python 3.13 for 3D workloads

## Installation Options

### Option 1: Use ComfyUI Embedded Python 3.11 (Recommended)

If you have ComfyUI with embedded Python 3.11:

```bash
# Navigate to your ComfyUI Python 3.11 installation
cd "D:\ComfyUi\Cuda-v12\python_embeded_3.11"

# Install PyTorch with CUDA 12.8 support
python.exe -m pip install torch>=2.9.0 torchvision>=0.24.0 torchaudio>=2.9.0 --index-url https://download.pytorch.org/whl/cu128

# Install RDAWG 3D Pack core dependencies
python.exe -m pip install trimesh>=4.0.0 open3d>=0.19.0 scipy>=1.11.0 matplotlib>=3.7.0

# Install RDAWG 3D Pack
cd ComfyUI\custom_nodes\rdawg-3d-pack
..\..\python_embeded_3.11\python.exe -m pip install -e .
```

### Option 2: Fresh Python 3.11 Installation

#### Windows

1. **Download Python 3.11**
   - Visit https://www.python.org/downloads/release/python-3119/
   - Download "Windows installer (64-bit)"

2. **Install Python 3.11**
   - Run installer as Administrator
   - Check "Add Python 3.11 to PATH"
   - Check "Install for all users"
   - Complete installation

3. **Install Dependencies**
   ```powershell
   # Upgrade pip
   python -m pip install --upgrade pip

   # Install PyTorch with CUDA 12.8
   pip install torch>=2.9.0 torchvision>=0.24.0 torchaudio>=2.9.0 --index-url https://download.pytorch.org/whl/cu128

   # Install RDAWG 3D Pack dependencies
   pip install trimesh>=4.0.0 open3d>=0.19.0 scipy>=1.11.0 matplotlib>=3.7.0 tqdm>=4.65.0 einops>=0.7.0
   ```

#### Linux

1. **Install Python 3.11**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3.11 python3.11-pip python3.11-venv

   # CentOS/RHEL/Fedora
   sudo dnf install python3.11 python3.11-pip
   ```

2. **Create Virtual Environment**
   ```bash
   python3.11 -m venv rdawg_3d_env
   source rdawg_3d_env/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install torch>=2.9.0 torchvision>=0.24.0 torchaudio>=2.9.0 --index-url https://download.pytorch.org/whl/cu128
   pip install trimesh>=4.0.0 open3d>=0.19.0 scipy>=1.11.0 matplotlib>=3.7.0
   ```

## Verification

### Test Python Version
```bash
python --version
# Should output: Python 3.11.x
```

### Test CUDA Support
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Test Open3D
```python
import open3d as o3d
print(f"Open3D version: {o3d.__version__}")

# Test basic functionality
mesh = o3d.geometry.TriangleMesh.create_sphere()
print(f"Created sphere with {len(mesh.vertices)} vertices")
```

### Test RDAWG 3D Pack
```bash
cd path/to/rdawg-3d-pack
python test_package.py
```

## Troubleshooting

### Common Issues

**Open3D Installation Fails**
```bash
# Use specific wheel for Python 3.11 Windows
pip install https://github.com/isl-org/Open3D/releases/download/v0.19.0/open3d-0.19.0-cp311-cp311-win_amd64.whl
```

**CUDA Not Available**
```bash
# Verify CUDA installation
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch>=2.9.0 torchvision>=0.24.0 torchaudio>=2.9.0 --index-url https://download.pytorch.org/whl/cu128
```

**Memory Issues**
```bash
# Set environment variable for memory allocation
export PYTORCH_ALLOC_CONF=max_split_size_mb:256
```

### Performance Optimization

**GPU Memory Management**
```python
import torch
# Set memory allocation strategy
torch.cuda.empty_cache()
```

**Multi-threading**
```python
import open3d as o3d
# Enable multi-threading
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
```

## Migration from Other Python Versions

### From Python 3.10
- Most packages are compatible
- Reinstall PyTorch for CUDA 12.8 support
- Open3D installation should work immediately

### From Python 3.12/3.13
- Open3D may not have pre-built wheels
- May need to compile from source (complex)
- Recommended to use Python 3.11 for stability

### From Python 3.9 or Earlier
- Upgrade recommended for better performance
- All dependencies support Python 3.11
- Significant performance improvements expected

## Next Steps

After successful Python 3.11 setup:

1. **Install RDAWG 3D Pack**:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/rdawgemfl/rdawg-3d-pack.git
   cd rdawg-3d-pack
   pip install -e .
   ```

2. **Run Tests**:
   ```bash
   python test_package.py
   ```

3. **Start ComfyUI** and look for "RDAWG 3D" nodes

4. **Check Examples** in the `examples/` directory

---

**💡 Tip**: Python 3.11 with Open3D 0.19.0 provides the best experience for RDAWG 3D Pack with maximum stability and performance.