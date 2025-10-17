# Installation Guide

## Quick Start

### Method 1: ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Search for `rdawg-3d-pack-cu128-pytorch-2.9.0`
3. Click Install
4. Restart ComfyUI

### Method 2: Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/rdawgemfl/rdawg-3d-pack.git
cd rdawg-3d-pack
python install.py
```

### Method 3: Development Installation
```bash
git clone https://github.com/rdawgemfl/rdawg-3d-pack.git
cd rdawg-3d-pack
python scripts/setup_development.py
```

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **PyTorch**: 2.9.0 or higher
- **CUDA**: 12.0 or higher (for GPU acceleration)
- **VRAM**: 4GB+ (8GB+ recommended for large models)
- **RAM**: 8GB+ (16GB+ recommended)

### Recommended Setup
- **Python**: 3.11 (best compatibility)
- **PyTorch**: 2.9.0+cu128
- **GPU**: RTX 30-series or newer
- **VRAM**: 16GB+
- **SSD Storage**

## Dependency Installation

### Core Dependencies (Required)
```bash
pip install torch>=2.9.0 torchvision>=0.24.0
pip install numpy>=1.24.0
pip install trimesh>=4.0.0
pip install tqdm>=4.65.0
pip install einops>=0.7.0
```

### Optional Dependencies

#### Full 3D Processing Suite
```bash
pip install open3d>=0.18.0
pip install pytorch3d>=0.7.5
pip install scipy>=1.11.0
pip install matplotlib>=3.7.0
```

#### Neural Rendering Capabilities
```bash
pip install transformers>=4.35.0
pip install diffusers>=0.24.0
pip install accelerate>=0.24.0
pip install xformers>=0.0.32
```

## Platform-Specific Installation

### Windows

#### Using Python Embedded (Recommended for ComfyUI)
```bash
# Navigate to your ComfyUI Python installation
cd "D:\ComfyUi\Cuda-v12\python_embeded"

# Install core dependencies
python.exe -m pip install torch>=2.9.0 torchvision>=0.24.0 --index-url https://download.pytorch.org/whl/cu128
python.exe -m pip install trimesh>=4.0.0 open3d>=0.18.0
```

#### Using System Python
```bash
# Install PyTorch with CUDA support
pip install torch>=2.9.0 torchvision>=0.24.0 torchaudio>=2.9.0 --index-url https://download.pytorch.org/whl/cu128

# Install 3D libraries
pip install trimesh open3d
```

### Linux
```bash
# Install PyTorch
pip install torch>=2.9.0 torchvision>=0.24.0 torchaudio>=2.9.0 --index-url https://download.pytorch.org/whl/cu128

# Install system dependencies for 3D processing
sudo apt-get update
sudo apt-get install libegl1-mesa-dev libgl1-mesa-dev

# Install Python packages
pip install trimesh open3d pytorch3d
```

## Troubleshooting

### Common Issues

#### PyTorch CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Reinstall PyTorch if needed
pip uninstall torch torchvision torchaudio
pip install torch>=2.9.0 torchvision>=0.24.0 --index-url https://download.pytorch.org/whl/cu128
```

#### Open3D Installation Issues
```bash
# For Windows: use pre-built wheel
python -m pip install https://github.com/isl-org/Open3D/releases/download/v0.19.0/open3d-0.19.0-cp311-cp311-win_amd64.whl

# For Linux: build from source if wheel fails
pip install open3d --verbose
```

#### Memory Issues
- Reduce model complexity for large meshes
- Use CPU mode if VRAM is limited
- Enable memory optimization in settings

#### Import Errors
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall package
pip uninstall rdawg-3d-pack-cu128-pytorch-2.9.0
pip install -e .
```

### Performance Optimization

#### GPU Memory Management
```python
# In your ComfyUI settings or environment variables
import os
os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:128'
```

#### Multi-GPU Setup
```python
# Specify GPU device
import torch
torch.cuda.set_device(0)  # Use first GPU
```

## Verification

### Test Installation
```bash
# Run the built-in test suite
python scripts/test.py

# Or test manually
python -c "
import torch
import trimesh
try:
    import open3d as o3d
    print('✅ Open3D available')
except ImportError:
    print('⚠️  Open3D not available')
print('✅ Installation successful!')
"
```

### Test in ComfyUI
1. Start ComfyUI
2. Look for "RDAWG 3D" nodes in the node menu
3. Try loading a simple 3D model
4. Check console for loading messages

## Configuration

### Environment Variables
```bash
# Optional: Set default device
export RDAWG_3D_DEVICE=cuda  # or cpu

# Optional: Memory optimization
export PYTORCH_ALLOC_CONF=max_split_size_mb:256
```

### ComfyUI Settings
Add to your ComfyUI configuration:
```json
{
  "rdawg_3d_pack": {
    "default_device": "auto",
    "memory_optimization": true,
    "open3d_priority": true
  }
}
```

## Next Steps

After successful installation:

1. **Browse Examples**: Check the `examples/` directory for workflow examples
2. **Read Documentation**: See `docs/USAGE.md` for detailed usage instructions
3. **Join Community**: Get support and share your creations
4. **Contribute**: Help improve the package by reporting issues or submitting pull requests

## Support

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check `docs/` for detailed guides
- **Community**: Join discussions and get help