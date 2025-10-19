"""
RDAWG 3D Pack - Modern 3D Processing Nodes for ComfyUI
Optimized for CUDA 12.8 + PyTorch 2.9.0
REQUIRES Open3D 0.19.0+
"""

import torch
import numpy as np
from typing import Tuple, Optional, List, Any
import os

# Auto-install dependencies on first import
try:
    from . import auto_install
except ImportError:
    # Fallback for direct execution
    pass

# Simple device management without ComfyUI dependency
def get_torch_device():
    """Get the best available torch device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# Import required libraries
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
    print(f"[RDAWG 3D Pack] Open3D {o3d.__version__} loaded successfully")
except ImportError:
    OPEN3D_AVAILABLE = False
    print("[RDAWG 3D Pack] ❌ ERROR: Open3D is required but not installed!")
    print("[RDAWG 3D Pack] Please run: pip install open3d>=0.19.0")
    raise ImportError("Open3D 0.19.0+ is required for RDAWG 3D Pack")

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("[RDAWG 3D Pack] Trimesh not available (optional fallback)")

class RDAWG3DLoadModel:
    """Load 3D models from various formats with GPU acceleration"""
    
    @classmethod
    def get_models_directory(cls):
        """Get the directory where 3D models are stored"""
        try:
            import folder_paths
            if hasattr(folder_paths, 'get_folder_paths'):
                try:
                    paths = folder_paths.get_folder_paths("3d_models")
                    if paths:
                        return paths[0]
                except:
                    pass
            return os.path.join(folder_paths.get_input_directory(), "3d_models")
        except:
            comfy_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            return os.path.join(comfy_dir, "input", "3d_models")
    
    @classmethod
    def get_available_models(cls):
        """Get list of available 3D model files"""
        models_dir = cls.get_models_directory()
        if not os.path.exists(models_dir):
            try:
                os.makedirs(models_dir, exist_ok=True)
            except:
                pass
            return ["No models found - place files in input/3d_models/"]
        
        supported_extensions = ['.obj', '.ply', '.stl', '.off', '.gltf', '.glb', '.fbx', '.dae', '.3ds']
        models = []
        try:
            for root, dirs, files in os.walk(models_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in supported_extensions):
                        rel_path = os.path.relpath(os.path.join(root, file), models_dir)
                        models.append(rel_path)
        except Exception as e:
            print(f"[RDAWG 3D Pack] Error scanning models directory: {e}")
        
        return sorted(models) if models else ["No models found - place files in input/3d_models/"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_file": (cls.get_available_models(), ),
                "load_texture": ("BOOLEAN", {"default": True}),
                "normalize": ("BOOLEAN", {"default": True}),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
              }
        }

    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("mesh", "info")
    FUNCTION = "load_3d_model"
    CATEGORY = "RDAWG 3D/Loaders"
    
    @classmethod
    def IS_CHANGED(cls, model_file, **kwargs):
        """Force refresh when model file changes"""
        models_dir = cls.get_models_directory()
        file_path = os.path.join(models_dir, model_file)
        if os.path.exists(file_path):
            return os.path.getmtime(file_path)
        return float("nan")

    def load_3d_model(self, model_file: str, load_texture: bool, normalize: bool, device: str):
        models_dir = self.get_models_directory()
        file_path = os.path.join(models_dir, model_file)
        
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}
Place 3D models in: {models_dir}")

        try:
            device_to_use = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
            if device == "auto":
                device_to_use = get_torch_device().type

            # Use Open3D (required)
            if not OPEN3D_AVAILABLE:
                raise RuntimeError("Open3D is required but not available. Please install Open3D 0.19.0+")

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
              }
        }

    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("mesh", "info")
    FUNCTION = "create_mesh"
    CATEGORY = "RDAWG 3D/Create"

    def create_mesh(self, vertices: torch.Tensor, faces: torch.Tensor, device: str,
                   vertex_colors: Optional[torch.Tensor] = None,
                   texture_uv: Optional[torch.Tensor] = None):

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

        # Create Open3D mesh (required)
        if not OPEN3D_AVAILABLE:
            raise RuntimeError("Open3D is required but not available. Please install Open3D 0.19.0+")

        try:
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices.cpu().numpy())
            o3d_mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
            o3d_mesh.compute_vertex_normals()
            mesh_data['open3d_mesh'] = o3d_mesh
            mesh_data['format'] = 'open3d_created'
        except Exception as e:
            raise RuntimeError(f"Open3D mesh creation failed: {e}")

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

        # Update Open3D mesh (required)
        if 'open3d_mesh' in transformed_mesh:
            try:
                o3d_mesh = transformed_mesh['open3d_mesh']
                o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices.cpu().numpy())
                o3d_mesh.compute_vertex_normals()
            except Exception as e:
                raise RuntimeError(f"Open3D mesh transform failed: {e}")

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
                           mesh_color: str, camera_distance: float, elevation: float, azimuth: float):

        # Use Open3D rendering (required)
        if not OPEN3D_AVAILABLE:
            raise RuntimeError("Open3D is required but not available. Please install Open3D 0.19.0+")

        if 'open3d_mesh' not in mesh:
            raise RuntimeError("Mesh must have Open3D mesh data for rendering")

        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors

            # Create visualizer
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=width, height=height, visible=False)
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
            raise RuntimeError(f"Open3D rendering failed: {e}")

# Import additional node modules
import importlib.util
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# Load mesh_processing nodes
try:
    mesh_processing_path = os.path.join(current_dir, 'mesh_processing.py')
    spec = importlib.util.spec_from_file_location("mesh_processing", mesh_processing_path)
    mesh_processing_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mesh_processing_module)
    MESH_PROCESSING_NODES = mesh_processing_module.NODE_CLASS_MAPPINGS
    MESH_PROCESSING_DISPLAY_NAMES = mesh_processing_module.NODE_DISPLAY_NAME_MAPPINGS
    print(f"[RDAWG 3D Pack] Loaded {len(MESH_PROCESSING_NODES)} mesh processing nodes")
except Exception as e:
    print(f"[RDAWG 3D Pack] Warning: Could not load mesh_processing: {e}")
    MESH_PROCESSING_NODES = {}
    MESH_PROCESSING_DISPLAY_NAMES = {}

# Load point_cloud nodes
try:
    point_cloud_path = os.path.join(current_dir, 'point_cloud.py')
    spec = importlib.util.spec_from_file_location("point_cloud", point_cloud_path)
    point_cloud_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(point_cloud_module)
    POINT_CLOUD_NODES = point_cloud_module.NODE_CLASS_MAPPINGS
    POINT_CLOUD_DISPLAY_NAMES = point_cloud_module.NODE_DISPLAY_NAME_MAPPINGS
    print(f"[RDAWG 3D Pack] Loaded {len(POINT_CLOUD_NODES)} point cloud nodes")
except Exception as e:
    print(f"[RDAWG 3D Pack] Warning: Could not load point_cloud: {e}")
    POINT_CLOUD_NODES = {}
    POINT_CLOUD_DISPLAY_NAMES = {}

# Load mesh_analysis nodes
try:
    mesh_analysis_path = os.path.join(current_dir, 'mesh_analysis.py')
    spec = importlib.util.spec_from_file_location("mesh_analysis", mesh_analysis_path)
    mesh_analysis_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mesh_analysis_module)
    MESH_ANALYSIS_NODES = mesh_analysis_module.NODE_CLASS_MAPPINGS
    MESH_ANALYSIS_DISPLAY_NAMES = mesh_analysis_module.NODE_DISPLAY_NAME_MAPPINGS
    print(f"[RDAWG 3D Pack] Loaded {len(MESH_ANALYSIS_NODES)} mesh analysis nodes")
except Exception as e:
    print(f"[RDAWG 3D Pack] Warning: Could not load mesh_analysis: {e}")
    MESH_ANALYSIS_NODES = {}
    MESH_ANALYSIS_DISPLAY_NAMES = {}

# Core node mappings
NODE_CLASS_MAPPINGS = {
    "RDAWG3DLoadModel": RDAWG3DLoadModel,
    "RDAWG3DCreateMesh": RDAWG3DCreateMesh,
    "RDAWG3DTransform": RDAWG3DTransform,
    "RDAWG3DMeshToImage": RDAWG3DMeshToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RDAWG3DLoadModel": "🔷 Load 3D Model (RDAWG+Open3D)",
    "RDAWG3DCreateMesh": "🔷 Create 3D Mesh (RDAWG+Open3D)",
    "RDAWG3DTransform": "🔷 Transform 3D Mesh (RDAWG)",
    "RDAWG3DMeshToImage": "🔷 3D to Image (RDAWG+Open3D)",
}

# Merge all node mappings
NODE_CLASS_MAPPINGS.update(MESH_PROCESSING_NODES)
NODE_CLASS_MAPPINGS.update(POINT_CLOUD_NODES)
NODE_CLASS_MAPPINGS.update(MESH_ANALYSIS_NODES)

NODE_DISPLAY_NAME_MAPPINGS.update(MESH_PROCESSING_DISPLAY_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(POINT_CLOUD_DISPLAY_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(MESH_ANALYSIS_DISPLAY_NAMES)

WEB_DIRECTORY = "./js"

print(f"[RDAWG 3D Pack] Loaded {len(NODE_CLASS_MAPPINGS)} nodes")
print(f"[RDAWG 3D Pack] REQUIRES Open3D {o3d.__version__}")
print(f"[RDAWG 3D Pack] CUDA 12.8 + PyTorch 2.9.0 Optimized")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']