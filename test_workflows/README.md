# RDAWG 3D Pack - Complete Test Suite

This directory contains comprehensive test workflows for all 19 RDAWG 3D Pack nodes, ensuring proper functionality and integration testing.

## 📁 Test Files Structure

```
test_workflows/
├── README.md                    # This file
├── core_nodes_test.json         # Test all 4 core nodes
├── mesh_processing_test.json    # Test all 5 mesh processing nodes
├── point_cloud_test.json        # Test all 6 point cloud nodes
├── mesh_analysis_test.json       # Test all 4 analysis nodes
├── complete_pipeline_test.json  # Full integration test
└── ../test_models/              # Test 3D models and point clouds
```

## 🧪 Test Categories

### 1. Core Nodes Test (`core_nodes_test.json`)
**Tests all 4 core RDAWG 3D nodes:**
- `RDAWG3DLoadModel` - Load 3D models from files
- `RDAWG3DCreateMesh` - Create meshes from vertices/faces
- `RDAWG3DTransform` - Apply transformations
- `RDAWG3DMeshToImage` - Render 3D to 2D images

**Features:**
- Tests model loading with different file formats
- Tests mesh creation from basic geometry
- Tests transformation operations (scale, rotation, translation)
- Tests rendering with different colors and camera angles

### 2. Mesh Processing Test (`mesh_processing_test.json`)
**Tests all 5 mesh processing nodes:**
- `RDAWG3DSimplifyMesh` - Mesh decimation
- `RDAWG3DSubdivideMesh` - Mesh subdivision
- `RDAWG3DSmoothMesh` - Surface smoothing
- `RDAWG3DComputeCurvature` - Curvature analysis
- `RDAWG3DRemeshUniform` - Uniform remeshing

**Features:**
- Tests simplification with different algorithms (quadric)
- Tests subdivision with loop method
- Tests smoothing with Taubin algorithm
- Tests curvature computation
- Tests various mesh processing parameters

### 3. Point Cloud Test (`point_cloud_test.json`)
**Tests all 6 point cloud nodes:**
- `RDAWG3DMeshToPointCloud` - Convert mesh to point cloud
- `RDAWG3DLoadPointCloud` - Load point cloud files
- `RDAWG3DDownsamplePointCloud` - Point cloud downsampling
- `RDAWG3DRemoveOutliers` - Outlier removal
- `RDAWG3DPointCloudToMesh` - Point cloud to mesh reconstruction
- `RDAWG3DTransformPointCloud` - Point cloud transformations

**Features:**
- Tests mesh to point cloud conversion (Poisson sampling)
- Tests point cloud loading and processing
- Tests outlier removal (statistical method)
- Tests downsampling (voxel method)
- Tests mesh reconstruction (Poisson, alpha shape)
- Tests point cloud transformations

### 4. Mesh Analysis Test (`mesh_analysis_test.json`)
**Tests all 4 analysis nodes:**
- `RDAWG3DAnalyzeMesh` - Comprehensive mesh analysis
- `RDAWG3DBoundingBox` - Bounding box computation
- `RDAWG3DComputeDistance` - Distance metrics
- `RDAWG3DExtractFeatures` - Feature extraction

**Features:**
- Tests mesh quality analysis (volume, surface area, topology)
- Tests bounding box computation (axis-aligned, oriented)
- Tests distance computation between meshes
- Tests feature extraction (curvature, edge angles, etc.)

### 5. Complete Pipeline Test (`complete_pipeline_test.json`)
**Full integration test combining all 19 nodes:**

**Pipeline Flow:**
1. Load high-poly mesh → Simplify → Smooth
2. Convert to point cloud → Remove outliers → Downsample
3. Reconstruct mesh → Analyze quality → Extract features
4. Transform mesh → Compute distances → Render results
5. Test bounding boxes and visualization

**Features:**
- Tests complete end-to-end workflows
- Tests data flow between different node types
- Tests parameter optimization
- Tests rendering of processed results
- Tests all node interactions and compatibility

## 🎯 How to Use the Test Suite

### Prerequisites
1. Ensure RDAWG 3D Pack is properly installed (19 nodes loaded)
2. Test models should be generated using `create_test_models.py`
3. ComfyUI should be running with GPU support

### Running Tests

1. **Load Test Workflow:**
   - Open ComfyUI
   - Click "Load" button
   - Navigate to `custom_nodes/rdawg_3D_pack/test_workflows/`
   - Select desired test JSON file

2. **Execute Test:**
   - Click "Queue Prompt" button
   - Monitor console for any errors
   - Check output images in your ComfyUI output folder

3. **Verify Results:**
   - All workflows should complete without errors
   - Output images should be generated in the output folder
   - Console should show successful node execution

### Expected Outputs

Each test workflow generates multiple output images:
- `RDAWG_core_test_*.png` - Core functionality results
- `RDAWG_simplified_mesh.png` - Mesh simplification result
- `RDAWG_subdivided_mesh.png` - Mesh subdivision result
- `RDAWG_smoothed_mesh.png` - Mesh smoothing result
- `RDAWG_pointcloud_reconstruction.png` - Point cloud reconstruction
- `RDAWG_pointcloud_cleaned.png` - Cleaned point cloud result
- `RDAWG_analysis_original.png` - Original mesh analysis
- `RDAWG_analysis_bbox.png` - Bounding box visualization
- `RDAWG_pipeline_*.png` - Complete pipeline results

## 🔍 Test Coverage

### Node Coverage (19/19 = 100%)
✅ Core Nodes (4/4)
- RDAWG3DLoadModel
- RDAWG3DCreateMesh
- RDAWG3DTransform
- RDAWG3DMeshToImage

✅ Mesh Processing (5/5)
- RDAWG3DSimplifyMesh
- RDAWG3DSubdivideMesh
- RDAWG3DSmoothMesh
- RDAWG3DRemeshUniform
- RDAWG3DComputeCurvature

✅ Point Cloud (6/6)
- RDAWG3DMeshToPointCloud
- RDAWG3DLoadPointCloud
- RDAWG3DDownsamplePointCloud
- RDAWG3DRemoveOutliers
- RDAWG3DPointCloudToMesh
- RDAWG3DTransformPointCloud

✅ Analysis (4/4)
- RDAWG3DAnalyzeMesh
- RDAWG3DComputeDistance
- RDAWG3DBoundingBox
- RDAWG3DExtractFeatures

### Feature Coverage
- ✅ File I/O operations (OBJ, PLY)
- ✅ Mesh processing algorithms
- ✅ Point cloud operations
- ✅ Geometric analysis
- ✅ 3D rendering
- ✅ Data transformations
- ✅ Error handling
- ✅ GPU acceleration

## 🚨 Troubleshooting

### Common Issues

1. **"File not found" errors:**
   - Ensure test models exist in `test_models/` directory
   - Run `create_test_models.py` to regenerate test files

2. **"Node not found" errors:**
   - Verify all 19 RDAWG 3D nodes are loaded in ComfyUI
   - Restart ComfyUI if nodes are missing

3. **GPU memory errors:**
   - Reduce image resolution in test workflows
   - Close other GPU-intensive applications

4. **Import errors:**
   - Ensure Open3D 0.19.0+ is installed
   - Check Python path configuration

### Debug Mode

To enable verbose logging:
1. Open ComfyUI with `--verbose` flag
2. Monitor console output for detailed node execution
3. Check for any warning or error messages

## 📈 Performance Metrics

Expected processing times (approximate):
- Core operations: 1-3 seconds
- Mesh processing: 2-10 seconds (depends on mesh complexity)
- Point cloud operations: 1-5 seconds
- Analysis operations: 1-2 seconds
- Rendering: 0.5-2 seconds per image

## 🔄 Continuous Testing

For development and debugging:
1. Modify test parameters to test edge cases
2. Add new test workflows for additional features
3. Update test models as needed
4. Monitor performance with different hardware configurations

## 📝 Test Report Template

When reporting test results, include:
- ComfyUI version
- RDAWG 3D Pack version
- GPU/CPU specifications
- Test completion status
- Any error messages
- Output quality assessment
- Performance observations

---

**Note:** This test suite is designed to work with the RDAWG 3D Pack v2.0+ and requires Open3D 0.19.0+ for full functionality.