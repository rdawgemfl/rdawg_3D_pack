"""
RDAWG 3D Pack - Modern 3D Processing Nodes for ComfyUI
Optimized for CUDA 12.8 + PyTorch 2.9.0
Enhanced with Open3D 0.19.0 support
"""

import torch
import numpy as np
from typing import Tuple, Optional, List, Any
import os

# Simple device management without ComfyUI dependency
def get_torch_device():
    """Get the best available torch device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# Try to import trimesh and Open3D
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("[RDAWG 3D Pack] Trimesh not available, using fallback mesh processing")

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
    print("[RDAWG 3D Pack] Open3D 0.19.0 loaded successfully")
except ImportError:
    OPEN3D_AVAILABLE = False
    print("[RDAWG 3D Pack] Open3D not available")

class RDAWG3DLoadModel:
    """Load 3D models from various formats with GPU acceleration"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "multiline": False}),
                "load_texture": ("BOOLEAN", {"default": True}),
                "normalize": ("BOOLEAN", {"default": True}),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
                "use_open3d": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("mesh", "info")
    FUNCTION = "load_3d_model"
    CATEGORY = "RDAWG 3D/Loaders"

    def load_3d_model(self, file_path: str, load_texture: bool, normalize: bool, device: str, use_open3d: bool):
        if not file_path or not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

        try:
            device_to_use = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
            if device == "auto":
                device_to_use = get_torch_device().type

            # Try Open3D first if available and requested
            if use_open3d and OPEN3D_AVAILABLE:
                try:
                    mesh = o3d.io.read_triangle_mesh(file_path)
                    if normalize and mesh.has_vertices():
                        mesh.compute_vertex_normals()
                        # Normalize to unit sphere
                        vertices = np.asarray(mesh.vertices)
                        center = vertices.mean(axis=0)
                        scale = np.max(np.linalg.norm(vertices - center, axis=1))
                        if scale > 0:
                            mesh.translate(-center)
                            mesh.scale(1.0/scale, center=(0,0,0))

                    mesh_data = {
                        'vertices': torch.from_numpy(np.asarray(mesh.vertices)).float().to(device_to_use),
                        'faces': torch.from_numpy(np.asarray(mesh.triangles)).long().to(device_to_use) if mesh.has_triangles() else None,
                        'vertex_normals': torch.from_numpy(np.asarray(mesh.vertex_normals)).float().to(device_to_use) if mesh.has_vertex_normals() else None,
                        'open3d_mesh': mesh,
                        'format': 'open3d'
                    }

                    info = f"Loaded (Open3D): {os.path.basename(file_path)}\\n"
                    info += f"Vertices: {len(mesh.vertices)}\\n"
                    info += f"Faces: {len(mesh.triangles) if mesh.has_triangles() else 0}\\n"
                    info += f"Device: {device_to_use}"

                    return (mesh_data, info)

                except Exception as e:
                    print(f"[RDAWG 3D Pack] Open3D loading failed: {e}, falling back to trimesh")

            # Fallback to trimesh
            if TRIMESH_AVAILABLE:
                mesh = trimesh.load(file_path, process=False)

                if normalize and hasattr(mesh, 'vertices'):
                    # Normalize to unit sphere
                    vertices = mesh.vertices - mesh.vertices.mean(axis=0)
                    scale = vertices.max()
                    vertices /= scale
                    mesh.vertices = vertices

                mesh_data = {
                    'vertices': torch.from_numpy(mesh.vertices).float().to(device_to_use),
                    'faces': torch.from_numpy(mesh.faces).long().to(device_to_use) if hasattr(mesh, 'faces') else None,
                    'texture_uv': torch.from_numpy(mesh.visual.uv).float().to(device_to_use) if hasattr(mesh.visual, 'uv') and load_texture else None,
                    'vertex_colors': torch.from_numpy(mesh.visual.vertex_colors).float().to(device_to_use) if hasattr(mesh.visual, 'vertex_colors') else None,
                    'face_colors': torch.from_numpy(mesh.visual.face_colors).float().to(device_to_use) if hasattr(mesh.visual, 'face_colors') else None,
                    'format': 'trimesh'
                }

                info = f"Loaded (Trimesh): {os.path.basename(file_path)}\\n"
                info += f"Vertices: {len(mesh.vertices)}\\n"
                info += f"Faces: {len(mesh.faces) if hasattr(mesh, 'faces') else 0}\\n"
                info += f"Device: {device_to_use}"

                return (mesh_data, info)
            else:
                raise RuntimeError("Neither Open3D nor Trimesh is available")

        except Exception as e:
            raise RuntimeError(f"Failed to load 3D model: {str(e)}")

class RDAWG3DCreateMesh:
    """Create 3D meshes from vertices and faces"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vertices": ("TENSOR",),
                "faces": ("TENSOR",),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
            },
            "optional": {
                "vertex_colors": ("TENSOR",),
                "texture_uv": ("TENSOR",),
                "create_open3d": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("mesh", "info")
    FUNCTION = "create_mesh"
    CATEGORY = "RDAWG 3D/Create"

    def create_mesh(self, vertices: torch.Tensor, faces: torch.Tensor, device: str,
                   vertex_colors: Optional[torch.Tensor] = None,
                   texture_uv: Optional[torch.Tensor] = None,
                   create_open3d: bool = True):

        device_to_use = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        if device == "auto":
            device_to_use = get_torch_device().type

        # Ensure tensors are on the correct device
        vertices = vertices.float().to(device_to_use)
        faces = faces.long().to(device_to_use)

        mesh_data = {
            'vertices': vertices,
            'faces': faces,
            'texture_uv': texture_uv.float().to(device_to_use) if texture_uv is not None else None,
            'vertex_colors': vertex_colors.float().to(device_to_use) if vertex_colors is not None else None,
            'face_colors': None,
            'format': 'created'
        }

        # Create Open3D mesh if requested and available
        if create_open3d and OPEN3D_AVAILABLE:
            try:
                o3d_mesh = o3d.geometry.TriangleMesh()
                o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices.cpu().numpy())
                o3d_mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
                o3d_mesh.compute_vertex_normals()
                mesh_data['open3d_mesh'] = o3d_mesh
                mesh_data['format'] = 'open3d_created'
            except Exception as e:
                print(f"[RDAWG 3D Pack] Open3D mesh creation failed: {e}")

        info = f"Created mesh\\n"
        info += f"Vertices: {vertices.shape[0]}\\n"
        info += f"Faces: {faces.shape[0]}\\n"
        info += f"Device: {device_to_use}"

        return (mesh_data, info)

class RDAWG3DTransform:
    """Apply transformations to 3D meshes"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "scale": ("FLOAT", {"default": 1.0, "step": 0.01}),
                "rotation_x": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "rotation_y": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "rotation_z": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "translate_x": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "translate_y": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "translate_z": ("FLOAT", {"default": 0.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("transformed_mesh",)
    FUNCTION = "transform_mesh"
    CATEGORY = "RDAWG 3D/Transform"

    def transform_mesh(self, mesh: dict, scale: float, rotation_x: float, rotation_y: float, rotation_z: float,
                      translate_x: float, translate_y: float, translate_z: float):

        vertices = mesh['vertices'].clone()

        # Apply scale
        if scale != 1.0:
            vertices *= scale

        # Apply rotation (convert degrees to radians)
        if rotation_x != 0.0 or rotation_y != 0.0 or rotation_z != 0.0:
            # Create rotation matrices
            rx = torch.tensor(rotation_x * np.pi / 180, device=vertices.device)
            ry = torch.tensor(rotation_y * np.pi / 180, device=vertices.device)
            rz = torch.tensor(rotation_z * np.pi / 180, device=vertices.device)

            # Rotation around X axis
            Rx = torch.tensor([[1, 0, 0],
                              [0, torch.cos(rx), -torch.sin(rx)],
                              [0, torch.sin(rx), torch.cos(rx)]], device=vertices.device)

            # Rotation around Y axis
            Ry = torch.tensor([[torch.cos(ry), 0, torch.sin(ry)],
                              [0, 1, 0],
                              [-torch.sin(ry), 0, torch.cos(ry)]], device=vertices.device)

            # Rotation around Z axis
            Rz = torch.tensor([[torch.cos(rz), -torch.sin(rz), 0],
                              [torch.sin(rz), torch.cos(rz), 0],
                              [0, 0, 1]], device=vertices.device)

            # Combined rotation
            R = Rz @ Ry @ Rx
            vertices = vertices @ R.T

        # Apply translation
        if translate_x != 0.0 or translate_y != 0.0 or translate_z != 0.0:
            translation = torch.tensor([translate_x, translate_y, translate_z], device=vertices.device)
            vertices += translation

        # Create transformed mesh
        transformed_mesh = mesh.copy()
        transformed_mesh['vertices'] = vertices

        # Update Open3D mesh if present
        if 'open3d_mesh' in transformed_mesh and OPEN3D_AVAILABLE:
            try:
                o3d_mesh = transformed_mesh['open3d_mesh']
                o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices.cpu().numpy())
                o3d_mesh.compute_vertex_normals()
            except Exception as e:
                print(f"[RDAWG 3D Pack] Open3D mesh transform failed: {e}")

        return (transformed_mesh,)

class RDAWG3DMeshToImage:
    """Render 3D mesh to 2D image using Open3D's advanced rendering"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "background_color": ("STRING", {"default": "#000000"}),
                "mesh_color": ("STRING", {"default": "#FFFFFF"}),
                "use_open3d_render": ("BOOLEAN", {"default": True}),
                "camera_distance": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.1}),
                "elevation": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0}),
                "azimuth": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rendered_image",)
    FUNCTION = "render_mesh_to_image"
    CATEGORY = "RDAWG 3D/Render"

    def render_mesh_to_image(self, mesh: dict, width: int, height: int, background_color: str,
                           mesh_color: str, use_open3d_render: bool, camera_distance: float,
                           elevation: float, azimuth: float):

        # Try Open3D rendering first
        if use_open3d_render and OPEN3D_AVAILABLE and 'open3d_mesh' in mesh:
            try:
                import matplotlib.pyplot as plt
                import matplotlib.colors as mcolors

                # Create visualizer
                vis = o3d.visualization.Visualizer()
                vis.create_window(width=width, height=height)
                vis.get_render_option().background_color = mcolors.to_rgb(background_color)
                vis.get_render_option().mesh_show_back_face = True

                # Add mesh
                o3d_mesh = mesh['open3d_mesh']
                if mesh_color != "#FFFFFF":
                    # Set mesh color
                    o3d_mesh.paint_uniform_color(mcolors.to_rgb(mesh_color))

                vis.add_geometry(o3d_mesh)

                # Set camera
                ctr = o3d_mesh.get_center()
                vis.get_view_control().set_front([0, 0, -1])
                vis.get_view_control().set_up([0, 1, 0])
                vis.get_view_control().set_zoom(1.0/camera_distance)
                vis.get_view_control().set_lookat(ctr)

                # Update and capture
                vis.poll_events()
                vis.update_renderer()
                vis.poll_events()

                # Capture image
                image = vis.capture_screen_float_buffer(do_render=True)
                vis.destroy_window()

                # Convert to torch tensor
                image_tensor = torch.from_numpy(np.array(image)).float()
                image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)

                return (image_tensor,)

            except Exception as e:
                print(f"[RDAWG 3D Pack] Open3D rendering failed: {e}, falling back to wireframe")

        # Fallback to simple wireframe rendering
        vertices = mesh['vertices'].cpu().numpy()
        faces = mesh['faces'].cpu().numpy() if mesh['faces'] is not None else None

        # Create a simple 2D projection
        elev_rad = np.radians(elevation)
        azim_rad = np.radians(azimuth)

        camera_x = camera_distance * np.cos(elev_rad) * np.sin(azim_rad)
        camera_y = camera_distance * np.sin(elev_rad)
        camera_z = camera_distance * np.cos(elev_rad) * np.cos(azim_rad)

        # Simple orthographic projection
        scale = min(width, height) * 0.4
        center_x, center_y = width // 2, height // 2

        # Project 3D to 2D
        projected_x = vertices[:, 0] * scale + center_x
        projected_y = -vertices[:, 1] * scale + center_y  # Flip Y axis
        projected_z = vertices[:, 2]

        # Create simple depth buffer
        image = np.zeros((height, width, 3), dtype=np.uint8)
        depth_buffer = np.full((height, width), -np.inf)

        # Parse colors
        def parse_color(color_str):
            color_str = color_str.lstrip('#')
            return tuple(int(color_str[i:i+2], 16) for i in (0, 2, 4))

        bg_color = parse_color(background_color)
        mesh_color = parse_color(mesh_color)

        # Fill background
        image[:] = bg_color

        # Simple wireframe rendering
        if faces is not None:
            for face in faces:
                for i in range(len(face)):
                    v1_idx, v2_idx = face[i], face[(i+1) % len(face)]
                    x1, y1, z1 = projected_x[v1_idx], projected_y[v1_idx], projected_z[v1_idx]
                    x2, y2, z2 = projected_x[v2_idx], projected_y[v2_idx], projected_z[v2_idx]

                    # Simple line drawing
                    steps = int(np.sqrt((x2-x1)**2 + (y2-y1)**2))
                    if steps > 0:
                        for t in range(steps + 1):
                            t_norm = t / steps
                            x = int(x1 + t_norm * (x2 - x1))
                            y = int(y1 + t_norm * (y2 - y1))
                            z = z1 + t_norm * (z2 - z1)

                            if 0 <= x < width and 0 <= y < height and z > depth_buffer[y, x]:
                                image[y, x] = mesh_color
                                depth_buffer[y, x] = z

        # Convert to torch tensor
        image_tensor = torch.from_numpy(image).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)

        return (image_tensor,)

# Import additional node modules
try:
    from . import mesh_analysis
    MESH_ANALYSIS_AVAILABLE = True
except ImportError:
    MESH_ANALYSIS_AVAILABLE = False
    print("[RDAWG 3D Pack] Mesh analysis module not available")

try:
    from . import mesh_processing
    MESH_PROCESSING_AVAILABLE = True
except ImportError:
    MESH_PROCESSING_AVAILABLE = False
    print("[RDAWG 3D Pack] Mesh processing module not available")

try:
    from . import point_cloud
    POINT_CLOUD_AVAILABLE = True
except ImportError:
    POINT_CLOUD_AVAILABLE = False
    print("[RDAWG 3D Pack] Point cloud module not available")

# Core node mappings
NODE_CLASS_MAPPINGS = {
    "RDAWG3DLoadModel": RDAWG3DLoadModel,
    "RDAWG3DCreateMesh": RDAWG3DCreateMesh,
    "RDAWG3DTransform": RDAWG3DTransform,
    "RDAWG3DMeshToImage": RDAWG3DMeshToImage,
}

# Add mesh analysis nodes if available
if MESH_ANALYSIS_AVAILABLE:
    NODE_CLASS_MAPPINGS.update({
        "RDAWG3DAnalyzeMesh": mesh_analysis.RDAWG3DAnalyzeMesh,
        "RDAWG3DComputeDistance": mesh_analysis.RDAWG3DComputeDistance,
        "RDAWG3DBoundingBox": mesh_analysis.RDAWG3DBoundingBox,
        "RDAWG3DExtractFeatures": mesh_analysis.RDAWG3DExtractFeatures,
    })

# Add mesh processing nodes if available
if MESH_PROCESSING_AVAILABLE:
    NODE_CLASS_MAPPINGS.update({
        "RDAWG3DSimplifyMesh": mesh_processing.RDAWG3DSimplifyMesh,
        "RDAWG3DSubdivideMesh": mesh_processing.RDAWG3DSubdivideMesh,
        "RDAWG3DSmoothMesh": mesh_processing.RDAWG3DSmoothMesh,
        "RDAWG3DRemeshUniform": mesh_processing.RDAWG3DRemeshUniform,
        "RDAWG3DComputeCurvature": mesh_processing.RDAWG3DComputeCurvature,
    })

# Add point cloud nodes if available
if POINT_CLOUD_AVAILABLE:
    NODE_CLASS_MAPPINGS.update({
        "RDAWG3DMeshToPointCloud": point_cloud.RDAWG3DMeshToPointCloud,
        "RDAWG3DLoadPointCloud": point_cloud.RDAWG3DLoadPointCloud,
        "RDAWG3DDownsamplePointCloud": point_cloud.RDAWG3DDownsamplePointCloud,
        "RDAWG3DRemoveOutliers": point_cloud.RDAWG3DRemoveOutliers,
        "RDAWG3DPointCloudToMesh": point_cloud.RDAWG3DPointCloudToMesh,
        "RDAWG3DTransformPointCloud": point_cloud.RDAWG3DTransformPointCloud,
    })

# Core display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "RDAWG3DLoadModel": "🔷 Load 3D Model (RDAWG+Open3D)",
    "RDAWG3DCreateMesh": "🔷 Create 3D Mesh (RDAWG+Open3D)",
    "RDAWG3DTransform": "🔷 Transform 3D Mesh (RDAWG)",
    "RDAWG3DMeshToImage": "🔷 3D to Image (RDAWG+Open3D)",
}

# Add mesh analysis display names if available
if MESH_ANALYSIS_AVAILABLE:
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "RDAWG3DAnalyzeMesh": "🔷 Analyze Mesh (RDAWG)",
        "RDAWG3DComputeDistance": "🔷 Compute Distance (RDAWG)",
        "RDAWG3DBoundingBox": "🔷 Bounding Box (RDAWG)",
        "RDAWG3DExtractFeatures": "🔷 Extract Features (RDAWG)",
    })

# Add mesh processing display names if available
if MESH_PROCESSING_AVAILABLE:
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "RDAWG3DSimplifyMesh": "🔷 Simplify Mesh (RDAWG)",
        "RDAWG3DSubdivideMesh": "🔷 Subdivide Mesh (RDAWG)",
        "RDAWG3DSmoothMesh": "🔷 Smooth Mesh (RDAWG)",
        "RDAWG3DRemeshUniform": "🔷 Remesh Uniform (RDAWG)",
        "RDAWG3DComputeCurvature": "🔷 Compute Curvature (RDAWG)",
    })

# Add point cloud display names if available
if POINT_CLOUD_AVAILABLE:
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "RDAWG3DMeshToPointCloud": "🔷 Mesh to Point Cloud (RDAWG)",
        "RDAWG3DLoadPointCloud": "🔷 Load Point Cloud (RDAWG)",
        "RDAWG3DDownsamplePointCloud": "🔷 Downsample Point Cloud (RDAWG)",
        "RDAWG3DRemoveOutliers": "🔷 Remove Outliers (RDAWG)",
        "RDAWG3DPointCloudToMesh": "🔷 Point Cloud to Mesh (RDAWG)",
        "RDAWG3DTransformPointCloud": "🔷 Transform Point Cloud (RDAWG)",
    })

WEB_DIRECTORY = "./js"

print(f"[RDAWG 3D Pack] Loaded {len(NODE_CLASS_MAPPINGS)} nodes (19 total)")
print(f"[RDAWG 3D Pack] Enhanced with Open3D {o3d.__version__ if OPEN3D_AVAILABLE else 'N/A'}")
print(f"[RDAWG 3D Pack] CUDA 12.8 + PyTorch 2.9.0 Optimized")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']