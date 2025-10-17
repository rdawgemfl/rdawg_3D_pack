"""
RDAWG 3D Pack - Core 3D Processing Nodes
Essential 3D operations with modern GPU acceleration
"""

import torch
import numpy as np
from typing import Tuple, Optional, List, Any
import os
import comfy.model_management as mm

# Try to import trimesh
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("[RDAWG 3D Pack] Trimesh not available, using fallback mesh processing")

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
            }
        }

    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("mesh", "info")
    FUNCTION = "load_3d_model"
    CATEGORY = "RDAWG 3D/Loaders"

    def load_3d_model(self, file_path: str, load_texture: bool, normalize: bool, device: str):
        if not file_path or not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

        try:
            # Use CPU for loading, then move to GPU if requested
            mesh = trimesh.load(file_path, process=False)

            if normalize and hasattr(mesh, 'vertices'):
                # Normalize to unit sphere
                vertices = mesh.vertices - mesh.vertices.mean(axis=0)
                scale = vertices.max()
                vertices /= scale
                mesh.vertices = vertices

            # Convert to torch tensors
            device_to_use = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
            if device == "auto":
                device_to_use = mm.get_torch_device().type

            mesh_data = {
                'vertices': torch.from_numpy(mesh.vertices).float().to(device_to_use),
                'faces': torch.from_numpy(mesh.faces).long().to(device_to_use) if hasattr(mesh, 'faces') else None,
                'texture_uv': torch.from_numpy(mesh.visual.uv).float().to(device_to_use) if hasattr(mesh.visual, 'uv') and load_texture else None,
                'vertex_colors': torch.from_numpy(mesh.visual.vertex_colors).float().to(device_to_use) if hasattr(mesh.visual, 'vertex_colors') else None,
                'face_colors': torch.from_numpy(mesh.visual.face_colors).float().to(device_to_use) if hasattr(mesh.visual, 'face_colors') else None,
            }

            info = f"Loaded: {os.path.basename(file_path)}\\n"
            info += f"Vertices: {len(mesh.vertices)}\\n"
            info += f"Faces: {len(mesh.faces) if hasattr(mesh, 'faces') else 0}\\n"
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
            device_to_use = mm.get_torch_device().type

        # Ensure tensors are on the correct device
        vertices = vertices.float().to(device_to_use)
        faces = faces.long().to(device_to_use)

        mesh_data = {
            'vertices': vertices,
            'faces': faces,
            'texture_uv': texture_uv.float().to(device_to_use) if texture_uv is not None else None,
            'vertex_colors': vertex_colors.float().to(device_to_use) if vertex_colors is not None else None,
            'face_colors': None,
        }

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

        return (transformed_mesh,)

class RDAWG3DMeshToImage:
    """Render 3D mesh to 2D image using modern rendering techniques"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "background_color": ("STRING", {"default": "#000000"}),
                "mesh_color": ("STRING", {"default": "#FFFFFF"}),
                "light_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),
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
                           mesh_color: str, light_intensity: float, camera_distance: float,
                           elevation: float, azimuth: float):

        # Simple orthographic projection rendering
        vertices = mesh['vertices'].cpu().numpy()
        faces = mesh['faces'].cpu().numpy() if mesh['faces'] is not None else None

        # Create a simple 2D projection
        # Convert spherical coordinates to Cartesian
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

NODE_CLASS_MAPPINGS = {
    "RDAWG3DLoadModel": RDAWG3DLoadModel,
    "RDAWG3DCreateMesh": RDAWG3DCreateMesh,
    "RDAWG3DTransform": RDAWG3DTransform,
    "RDAWG3DMeshToImage": RDAWG3DMeshToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RDAWG3DLoadModel": "🔷 Load 3D Model (RDAWG)",
    "RDAWG3DCreateMesh": "🔷 Create 3D Mesh (RDAWG)",
    "RDAWG3DTransform": "🔷 Transform 3D Mesh (RDAWG)",
    "RDAWG3DMeshToImage": "🔷 3D to Image (RDAWG)",
}