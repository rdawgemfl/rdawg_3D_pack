# RDAWG 3D Pack - CUDA 12.8 + PyTorch 2.9.0 Optimized

🔷 **Modern 3D Processing Nodes for ComfyUI**

A comprehensive, next-generation 3D processing toolkit specifically optimized for CUDA 12.8 and PyTorch 2.9.0. This custom node package provides advanced 3D model loading, processing, transformation, and rendering capabilities with GPU acceleration.

## ✨ Key Features

### 🚀 **Performance Optimized**
- **CUDA 12.8 Native Support** - Leverages latest CUDA features
- **PyTorch 2.9.0+ Compatible** - Built for modern PyTorch architecture
- **GPU Memory Efficient** - Smart memory management for large 3D models
- **Batch Processing Support** - Process multiple 3D models simultaneously

### 🔧 **Core Functionality**
- **3D Model Loading** - Support for OBJ, STL, PLY, GLTF formats
- **Mesh Processing** - Vertex/face manipulation, subdivision, smoothing
- **Point Cloud Operations** - Filtering, downsampling, clustering
- **3D Transformations** - Scale, rotate, translate with precision control
- **Neural Rendering** - AI-powered 3D to 2D conversion
- **File Format Conversion** - Comprehensive 3D format support

### 🎯 **Advanced Features**
- **GPU-Accelerated Rendering** - Real-time 3D visualization
- **Custom Shader Support** - GLSL shader integration
- **Batch Operations** - Process multiple meshes in parallel
- **Memory Management** - Automatic VRAM optimization
- **Error Recovery** - Robust error handling and fallbacks

## 📦 Installation

### Automatic Installation (Recommended)
1. Use ComfyUI Manager to install `rdawg-3d-pack-cu128-pytorch-2.9.0`
2. Restart ComfyUI

### Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/rdawgemfl/rdawg-3d-pack-cu128-pytorch-2.9.0
cd rdawg-3d-pack-cu128-pytorch-2.9.0
pip install -r requirements.txt
```

### Optional Dependencies
```bash
# Full 3D processing suite
pip install -e .[full]

# Neural rendering capabilities
pip install -e .[neural]

# 3D reconstruction tools
pip install -e .[reconstruction]
```

## 🎮 Usage

### Basic 3D Model Loading
1. Add **🔷 Load 3D Model (RDAWG)** node
2. Specify your 3D file path
3. Configure loading options (texture, normalization, device)
4. Connect to downstream processing nodes

### 3D Transformations
1. Load your 3D model
2. Add **🔷 Transform 3D Mesh (RDAWG)** node
3. Adjust scale, rotation, and translation parameters
4. Connect transformed mesh to rendering or export nodes

### 3D to 2D Rendering
1. Load or create a 3D mesh
2. Add **🔷 3D to Image (RDAWG)** node
3. Configure rendering parameters (resolution, lighting, camera)
4. Output rendered image to standard ComfyUI image nodes

## 🔧 Node Reference

### Core Nodes

| Node | Description | Inputs | Outputs |
|------|-------------|--------|---------|
| 🔷 Load 3D Model | Load 3D models from files | file_path, options | mesh, info |
| 🔷 Create 3D Mesh | Create mesh from tensors | vertices, faces | mesh, info |
| 🔷 Transform 3D Mesh | Apply transformations | mesh, transform params | transformed_mesh |
| 🔷 3D to Image | Render 3D to 2D | mesh, render settings | image |

### Advanced Nodes (Optional Dependencies)

| Node | Category | Description |
|------|----------|-------------|
| Point Cloud Filter | RDAWG 3D/PointCloud | Filter and process point clouds |
| Mesh Subdivision | RDAWG 3D/Mesh | Subdivide and smooth meshes |
| Neural Renderer | RDAWG 3D/Neural | AI-powered 3D rendering |
| Format Converter | RDAWG 3D/Utils | Convert between 3D formats |

## ⚙️ Configuration

### Device Selection
- **Auto** - Automatically select best device (recommended)
- **CUDA** - Force GPU acceleration
- **CPU** - Fallback CPU processing

### Memory Settings
- **Normal VRAM** - Standard memory usage
- **High VRAM** - Faster processing with more memory
- **Low VRAM** - Memory-constrained processing

### Performance Tips
1. **Use CUDA** for large models when possible
2. **Enable batch processing** for multiple objects
3. **Adjust mesh resolution** based on your needs
4. **Monitor VRAM usage** with complex scenes

## 🎯 Examples

### Basic 3D Model Viewer Workflow
```
Load 3D Model → Transform 3D Mesh → 3D to Image → Save Image
```

### Advanced 3D Processing
```
Load 3D Model → Point Cloud Filter → Mesh Subdivision → Neural Renderer → Enhanced Image
```

### Batch Processing
```
Load Multiple Models → Batch Transform → Batch Render → Image Sequence
```

## 🔧 Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Reduce model complexity
- Use CPU mode for preprocessing
- Enable memory optimization settings

**Loading Failures**
- Check file format compatibility
- Verify file paths
- Ensure sufficient disk space

**Performance Issues**
- Update GPU drivers
- Check CUDA installation
- Monitor system resources

### Dependencies

**Minimum Requirements:**
- CUDA 12.8 compatible GPU
- PyTorch 2.9.0+
- Python 3.8+
- 8GB+ VRAM (for large models)

**Recommended Setup (Optimal Configuration):**
- **Python 3.11** (best Open3D compatibility)
- RTX 30-series or newer GPU
- 16GB+ VRAM
- Latest GPU drivers
- SSD storage

**🎯 Why Python 3.11?**
- Perfect Open3D 0.19.0 wheel support
- Excellent PyTorch CUDA integration
- Most stable for 3D processing libraries
- Better performance than Python 3.13 for scientific computing

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Credits

- **PyTorch 3D** - Core 3D processing capabilities
- **Trimesh** - Mesh processing utilities
- **Open3D** - Point cloud operations
- **ComfyUI Community** - Framework and inspiration

## 📞 Support

- **GitHub Issues** - Report bugs and request features
- **Discord** - Community support and discussions
- **Documentation** - Comprehensive guides and tutorials

---

**RDAWG 3D Pack** - Pushing the boundaries of 3D processing in ComfyUI! 🚀