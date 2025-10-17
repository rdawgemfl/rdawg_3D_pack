# Usage Guide

## Getting Started

### Basic Workflow
The fundamental RDAWG 3D workflow consists of three main steps:

1. **Load 3D Model** - Import your 3D file
2. **Transform** - Apply modifications (scale, rotate, position)
3. **Render** - Convert to 2D image

```
Load 3D Model → Transform 3D Mesh → 3D to Image → Save Image
```

## Node Reference

### Core Nodes

#### 🔷 Load 3D Model (RDAWG+Open3D)
**Purpose**: Load 3D models from various file formats

**Inputs**:
- `file_path` (STRING): Path to 3D file (OBJ, STL, PLY, GLTF)
- `load_texture` (BOOLEAN): Load texture coordinates if available
- `normalize` (BOOLEAN): Normalize model to unit sphere
- `device` (OPTIONS): Processing device (auto/cpu/cuda)
- `use_open3d` (BOOLEAN): Prioritize Open3D over Trimesh

**Outputs**:
- `mesh` (MESH): Loaded 3D mesh data
- `info` (STRING): Loading information and statistics

**Tips**:
- Use "auto" device for best performance
- Enable normalization for consistent scaling
- Open3D provides better loading for complex formats

**Example Usage**:
```
file_path: "C:/models/car.obj"
load_texture: true
normalize: true
device: "auto"
use_open3d: true
```

#### 🔷 Create 3D Mesh (RDAWG+Open3D)
**Purpose**: Create 3D mesh from vertex and face tensors

**Inputs**:
- `vertices` (TENSOR): Vertex coordinates [N, 3]
- `faces` (TENSOR): Face indices [M, 3]
- `device` (OPTIONS): Processing device (auto/cpu/cuda)
- `vertex_colors` (TENSOR, optional): Vertex colors [N, 3/4]
- `texture_uv` (TENSOR, optional): UV coordinates [N, 2]
- `create_open3d` (BOOLEAN): Create Open3D mesh object

**Outputs**:
- `mesh` (MESH): Created 3D mesh
- `info` (STRING): Creation information

**Tips**:
- Vertices should be in XYZ format
- Faces should reference vertex indices
- Vertex colors support RGB or RGBA

#### 🔷 Transform 3D Mesh (RDAWG)
**Purpose**: Apply geometric transformations to 3D meshes

**Inputs**:
- `mesh` (MESH): Input mesh to transform
- `scale` (FLOAT): Uniform scale factor (1.0 = no change)
- `rotation_x` (FLOAT): Rotation around X axis in degrees
- `rotation_y` (FLOAT): Rotation around Y axis in degrees
- `rotation_z` (FLOAT): Rotation around Z axis in degrees
- `translate_x` (FLOAT): Translation along X axis
- `translate_y` (FLOAT): Translation along Y axis
- `translate_z` (FLOAT): Translation along Z axis

**Outputs**:
- `transformed_mesh` (MESH): Transformed mesh

**Tips**:
- Transformations are applied in order: scale → rotate → translate
- Use small rotation values for precise control
- Translation units match model units

**Common Transformations**:
```
Scale up 2x: scale = 2.0
Rotate 45° around Y: rotation_y = 45.0
Move up 1 unit: translate_y = 1.0
```

#### 🔷 3D to Image (RDAWG+Open3D)
**Purpose**: Render 3D mesh to 2D image

**Inputs**:
- `mesh` (MESH): Mesh to render
- `width` (INT): Output image width (64-2048)
- `height` (INT): Output image height (64-2048)
- `background_color` (STRING): Hex color for background (#RRGGBB)
- `mesh_color` (STRING): Hex color for mesh (#RRGGBB)
- `use_open3d_render` (BOOLEAN): Use Open3D advanced rendering
- `camera_distance` (FLOAT): Camera distance (0.5-10.0)
- `elevation` (FLOAT): Camera elevation angle (-90° to 90°)
- `azimuth` (FLOAT): Camera azimuth angle (-180° to 180°)

**Outputs**:
- `rendered_image` (IMAGE): 2D rendered image

**Tips**:
- Higher resolution = better quality but slower
- Open3D rendering provides better lighting and shading
- Camera controls work like a virtual camera orbit

**Camera Settings**:
```
Front view: elevation = 0, azimuth = 0
Top view: elevation = 90, azimuth = 0
Side view: elevation = 0, azimuth = 90
```

## Workflow Examples

### Basic Model Viewer
Create a simple 3D model viewer:

1. **Load 3D Model**
   - File path: your model file
   - Enable normalization and textures

2. **Transform 3D Mesh**
   - Set rotation_y to 45 for angled view
   - Adjust scale if needed

3. **3D to Image**
   - Set resolution to 1024x1024
   - Use white background, black mesh
   - Set camera distance to 2.0

4. **Save Image**
   - Connect rendered image to SaveImage node

### Product Photography Setup
Render product images from multiple angles:

1. **Load 3D Model** → Load your product model
2. **Transform 3D Mesh** → Apply product positioning
3. **3D to Image** → Render with studio lighting
4. **Save Image** → Export high-quality images

**Pro Tips**:
- Use neutral background (#FFFFFF or #F0F0F0)
- Set mesh_color to "#808080" for professional look
- Try different camera angles for best presentation

### Architectural Visualization
Render architectural models:

1. **Load 3D Model** → Load building model
2. **Transform 3D Mesh** → Position for best view
3. **3D to Image** → Render with appropriate lighting
4. **Save Image** → Export presentation images

**Settings for Architecture**:
- Camera distance: 3.0-5.0 (show more context)
- Elevation: 15-30° (natural eye level)
- Background: "#87CEEB" (sky blue)

## Advanced Techniques

### Batch Processing
Process multiple models or views:

1. Create multiple Transform nodes with different rotations
2. Use multiple 3D to Image nodes
3. Combine outputs for comparison or animation

### Material Variations
Create different material looks:

1. Load single model
2. Create multiple Transform nodes (same settings)
3. Use different mesh_color values in 3D to Image nodes
4. Compare material variations

### Size Comparison
Scale models for size comparison:

1. Load multiple models
2. Use Transform nodes to normalize sizes
3. Render with consistent camera settings
4. Create comparison grid

## Performance Optimization

### Memory Management
- **Large Models**: Reduce mesh complexity before loading
- **High Resolution**: Start with lower resolution, increase as needed
- **Batch Processing**: Process items sequentially to avoid memory overflow

### Rendering Speed
- **Open3D vs Wireframe**: Use wireframe for faster previews
- **Resolution**: Use 512x512 for testing, 1024x1024+ for final
- **Device**: Use CUDA for GPU acceleration when available

### Quality vs Speed
| Setting | Fast | High Quality |
|---------|------|--------------|
| Resolution | 512x512 | 2048x2048 |
| Rendering | Wireframe | Open3D |
| Anti-aliasing | Off | On |

## Troubleshooting

### Common Issues

**Model doesn't load**:
- Check file path and format
- Ensure file is not corrupted
- Try different format (OBJ, STL, PLY)

**Black screen on render**:
- Check mesh_color and background_color
- Verify camera distance (not too close/far)
- Ensure mesh has vertices and faces

**Memory errors**:
- Reduce resolution
- Simplify mesh geometry
- Use CPU instead of CUDA

**Poor image quality**:
- Increase resolution
- Enable Open3D rendering
- Adjust camera position

### Performance Tips

**Faster Loading**:
- Use simple file formats (OBJ, STL)
- Disable texture loading if not needed
- Use CPU for small models

**Better Rendering**:
- Use Open3D for realistic lighting
- Increase resolution for final output
- Experiment with camera angles

**Memory Efficiency**:
- Clear unused nodes
- Use appropriate resolution
- Monitor VRAM usage

## File Format Support

### Supported Formats
- **OBJ**: Wavefront OBJ (recommended)
- **STL**: Stereolithography (mesh only)
- **PLY**: Stanford PLY (with colors)
- **GLTF**: GL Transmission Format

### Format Recommendations
- **General Use**: OBJ (good compatibility)
- **3D Printing**: STL (simple mesh)
- **High Quality**: PLY (supports colors)
- **Web Ready**: GLTF (compressed)

### Conversion Tips
```python
# Convert STL to OBJ if needed
import trimesh
mesh = trimesh.load('model.stl')
mesh.export('model.obj')
```

## Integration with Other Nodes

### Image Processing
- Connect to **Image Resize** for different resolutions
- Use **Image Blend** to add backgrounds
- Apply **Image Filter** for post-processing

### Animation
- Use **KSampler** with varying rotation values
- Create frame sequences with **Image Batch**
- Export as GIF or video

### AI Integration
- Connect to **Upscale** nodes for enhanced quality
- Use **ControlNet** for guided generation
- Apply **Style Transfer** for artistic effects

## Best Practices

### Workflow Organization
- Group related nodes
- Use descriptive names
- Keep workflows modular

### File Management
- Use organized folder structure
- Keep original files separate
- Document custom settings

### Performance Monitoring
- Watch VRAM usage with complex models
- Monitor render times
- Optimize based on your hardware

### Backup and Versioning
- Save workflow versions
- Backup custom models
- Document successful settings