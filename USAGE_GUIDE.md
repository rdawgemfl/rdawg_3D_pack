# RDAWG 3D Pack - Usage Guide

## Loading 3D Models

The **Load 3D Model** node now uses a dropdown file picker for easy model selection.

### Setup

1. **Create the models directory** (if it doesn't exist):
   ```
   ComfyUI/input/3d_models/
   ```

2. **Place your 3D model files** in this directory
   - You can organize them in subdirectories
   - Supported formats: `.obj`, `.ply`, `.stl`, `.gltf`, `.glb`, `.fbx`, `.dae`, `.3ds`, `.off`

3. **Restart ComfyUI** to see the models in the dropdown

### Using the Load 3D Model Node

1. Add the **"Load 3D Model (RDAWG+Open3D)"** node to your workflow
2. Click the **model_file** dropdown
3. Select your 3D model from the list
4. Configure other parameters:
   - **load_texture**: Load texture data if available
   - **normalize**: Normalize model to unit sphere
   - **device**: Choose CPU, CUDA, or auto

### Example Directory Structure

```
ComfyUI/input/3d_models/
├── cube.obj
├── sphere.ply
├── characters/
│   ├── character1.glb
│   └── character2.obj
└── props/
    ├── prop1.stl
    └── prop2.ply
```

Files in subdirectories will appear as `characters/character1.glb` in the dropdown.

### Supported File Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| Wavefront OBJ | `.obj` | Common 3D format with texture support |
| Polygon File Format | `.ply` | Point cloud and mesh format |
| Stereolithography | `.stl` | 3D printing format |
| Object File Format | `.off` | Simple mesh format |
| GL Transmission Format | `.gltf`, `.glb` | Modern 3D format |
| Filmbox | `.fbx` | Autodesk format |
| COLLADA | `.dae` | XML-based 3D format |
| 3D Studio | `.3ds` | Legacy 3D format |

### Troubleshooting

**Models not showing in dropdown:**
- Verify files are in `ComfyUI/input/3d_models/`
- Check file extensions are supported
- Restart ComfyUI completely
- Check console for error messages

**"No models found" message:**
- Place at least one 3D model file in the directory
- Ensure file has a supported extension
- Check file permissions

**File loading errors:**
- Verify the 3D model file is not corrupted
- Try opening the file in a 3D viewer (Blender, MeshLab, etc.)
- Check that Open3D supports the specific file variant

### Getting Sample Models

Free 3D models can be found at:
- [Sketchfab](https://sketchfab.com/) - Many free models
- [TurboSquid](https://www.turbosquid.com/Search/3D-Models/free) - Free section
- [Free3D](https://free3d.com/) - Free models
- [CGTrader](https://www.cgtrader.com/free-3d-models) - Free models

**Note:** Always respect the licenses of downloaded models!

## Technical Details

### How It Works

The node scans the `input/3d_models/` directory at startup and creates a dropdown list of all supported 3D model files. When you select a model, it constructs the full path and loads it using Open3D.

### Key Methods

- **`get_models_directory()`**: Locates the ComfyUI input/3d_models directory
- **`get_available_models()`**: Scans directory and returns list of model files
- **`IS_CHANGED()`**: Detects when model files are modified for cache invalidation

### Migration from Old Version

If you were using the old text input version:

**Old way:**
```
file_path: "C:/full/path/to/model.obj"
```

**New way:**
1. Move your model to `ComfyUI/input/3d_models/`
2. Select from dropdown: `model.obj`

The new method is more user-friendly and prevents path errors!

