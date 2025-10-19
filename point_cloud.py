"""
Point Cloud Processing Nodes for RDAWG 3D Pack
Utilizes Open3D's comprehensive point cloud capabilities
"""

import torch
import numpy as np
import open3d as o3d
from typing import Optional, Tuple, Any
import copy
import os

class RDAWG3DMeshToPointCloud:
    """Convert mesh to point cloud by sampling surface points"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "num_points": ("INT", {"default": 10000, "min": 100, "max": 100000}),
                "include_normals": ("BOOLEAN", {"default": True}),
                "include_colors": ("BOOLEAN", {"default": False}),
                "method": (["uniform", "poisson"], {"default": "uniform"}),
            }
        }

    RETURN_TYPES = ("POINT_CLOUD", "STRING")
    RETURN_NAMES = ("point_cloud", "info")
    FUNCTION = "mesh_to_point_cloud"
    CATEGORY = "RDAWG 3D/Point Cloud"

    def mesh_to_point_cloud(self, mesh: dict, num_points: int, include_normals: bool,
                          include_colors: bool, method: str):
        if 'open3d_mesh' not in mesh:
            raise ValueError("Mesh must have Open3D mesh data")

        o3d_mesh = mesh['open3d_mesh']

        try:
            if method == "uniform":
                # Uniform sampling
                pcd = o3d_mesh.sample_points_uniformly(number_of_points=num_points)
            else:
                # Poisson disk sampling
                pcd = o3d_mesh.sample_points_poisson_disk(number_of_points=num_points)

            # Ensure normals are computed if requested
            if include_normals and not pcd.has_normals():
                pcd.estimate_normals()

            # Create point cloud data structure
            device = mesh['vertices'].device
            point_cloud_data = {
                'points': torch.from_numpy(np.asarray(pcd.points)).float().to(device),
                'open3d_pcd': pcd,
                'format': 'open3d_point_cloud'
            }

            if include_normals and pcd.has_normals():
                point_cloud_data['normals'] = torch.from_numpy(np.asarray(pcd.normals)).float().to(device)

            if include_colors and pcd.has_colors():
                point_cloud_data['colors'] = torch.from_numpy(np.asarray(pcd.colors)).float().to(device)

            info = f"Converted mesh to point cloud\n"
            info += f"Method: {method}\n"
            info += f"Points: {len(pcd.points)}\n"
            info += f"Has normals: {pcd.has_normals()}\n"
            info += f"Has colors: {pcd.has_colors()}"

            return (point_cloud_data, info)

        except Exception as e:
            raise RuntimeError(f"Mesh to point cloud conversion failed: {e}")

class RDAWG3DLoadPointCloud:
    """Load point cloud from various formats"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "multiline": False}),
                "estimate_normals": ("BOOLEAN", {"default": True}),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("POINT_CLOUD", "STRING")
    RETURN_NAMES = ("point_cloud", "info")
    FUNCTION = "load_point_cloud"
    CATEGORY = "RDAWG 3D/Point Cloud"

    def load_point_cloud(self, file_path: str, estimate_normals: bool, device: str):
        if not file_path or not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

        try:
            # Load point cloud using Open3D
            pcd = o3d.io.read_point_cloud(file_path)

            if len(pcd.points) == 0:
                raise ValueError("No points found in file")

            # Estimate normals if requested
            if estimate_normals and not pcd.has_normals():
                pcd.estimate_normals()

            # Determine device
            if device == "auto":
                device_to_use = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device_to_use = device

            # Create point cloud data structure
            point_cloud_data = {
                'points': torch.from_numpy(np.asarray(pcd.points)).float().to(device_to_use),
                'open3d_pcd': pcd,
                'format': 'open3d_point_cloud'
            }

            if pcd.has_normals():
                point_cloud_data['normals'] = torch.from_numpy(np.asarray(pcd.normals)).float().to(device_to_use)

            if pcd.has_colors():
                point_cloud_data['colors'] = torch.from_numpy(np.asarray(pcd.colors)).float().to(device_to_use)

            info = f"Loaded point cloud\n"
            info += f"File: {os.path.basename(file_path)}\n"
            info += f"Points: {len(pcd.points)}\n"
            info += f"Has normals: {pcd.has_normals()}\n"
            info += f"Has colors: {pcd.has_colors()}\n"
            info += f"Device: {device_to_use}"

            return (point_cloud_data, info)

        except Exception as e:
            raise RuntimeError(f"Failed to load point cloud: {e}")

class RDAWG3DDownsamplePointCloud:
    """Downsample point cloud using various methods"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "point_cloud": ("POINT_CLOUD",),
                "method": (["voxel", "uniform", "random"], {"default": "voxel"}),
                "voxel_size": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 1.0, "step": 0.001}),
                "target_points": ("INT", {"default": 1000, "min": 10, "max": 100000}),
            }
        }

    RETURN_TYPES = ("POINT_CLOUD", "STRING")
    RETURN_NAMES = ("downsampled_pcd", "info")
    FUNCTION = "downsample_point_cloud"
    CATEGORY = "RDAWG 3D/Point Cloud"

    def downsample_point_cloud(self, point_cloud: dict, method: str, voxel_size: float, target_points: int):
        if 'open3d_pcd' not in point_cloud:
            raise ValueError("Point cloud must have Open3D data")

        pcd = copy.deepcopy(point_cloud['open3d_pcd'])

        try:
            if method == "voxel":
                downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
            elif method == "uniform":
                downsampled = pcd.uniform_down_sample(every_k_points=max(1, len(pcd.points) // target_points))
            else:  # random
                indices = np.random.choice(len(pcd.points), min(target_points, len(pcd.points)), replace=False)
                downsampled = pcd.select_by_index(indices)

            # Re-estimate normals if they existed
            if pcd.has_normals():
                downsampled.estimate_normals()

            # Create new point cloud data
            new_point_cloud = point_cloud.copy()
            new_point_cloud['open3d_pcd'] = downsampled
            new_point_cloud['points'] = torch.from_numpy(np.asarray(downsampled.points)).float().to(point_cloud['points'].device)

            if downsampled.has_normals():
                new_point_cloud['normals'] = torch.from_numpy(np.asarray(downsampled.normals)).float().to(point_cloud['points'].device)

            if downsampled.has_colors():
                new_point_cloud['colors'] = torch.from_numpy(np.asarray(downsampled.colors)).float().to(point_cloud['points'].device)

            info = f"Downsampled point cloud using {method}\n"
            info += f"Original points: {len(pcd.points)}\n"
            info += f"Downsampled points: {len(downsampled.points)}\n"
            info += f"Reduction: {(1 - len(downsampled.points)/len(pcd.points))*100:.1f}%"

            return (new_point_cloud, info)

        except Exception as e:
            raise RuntimeError(f"Point cloud downsampling failed: {e}")

class RDAWG3DRemoveOutliers:
    """Remove outliers from point cloud"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "point_cloud": ("POINT_CLOUD",),
                "method": (["statistical", "radius"], {"default": "statistical"}),
                "nb_neighbors": ("INT", {"default": 20, "min": 5, "max": 100}),
                "std_ratio": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "radius": ("FLOAT", {"default": 0.05, "min": 0.001, "max": 1.0, "step": 0.001}),
                "min_points": ("INT", {"default": 10, "min": 2, "max": 100}),
            }
        }

    RETURN_TYPES = ("POINT_CLOUD", "STRING")
    RETURN_NAMES = ("cleaned_pcd", "info")
    FUNCTION = "remove_outliers"
    CATEGORY = "RDAWG 3D/Point Cloud"

    def remove_outliers(self, point_cloud: dict, method: str, nb_neighbors: int, std_ratio: float,
                       radius: float, min_points: int):
        if 'open3d_pcd' not in point_cloud:
            raise ValueError("Point cloud must have Open3D data")

        pcd = copy.deepcopy(point_cloud['open3d_pcd'])

        try:
            if method == "statistical":
                cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
            else:  # radius
                cl, ind = pcd.remove_radius_outlier(nb_points=min_points, radius=radius)

            cleaned = pcd.select_by_index(ind)

            # Re-estimate normals if they existed
            if pcd.has_normals():
                cleaned.estimate_normals()

            # Create new point cloud data
            new_point_cloud = point_cloud.copy()
            new_point_cloud['open3d_pcd'] = cleaned
            new_point_cloud['points'] = torch.from_numpy(np.asarray(cleaned.points)).float().to(point_cloud['points'].device)

            if cleaned.has_normals():
                new_point_cloud['normals'] = torch.from_numpy(np.asarray(cleaned.normals)).float().to(point_cloud['points'].device)

            if cleaned.has_colors():
                new_point_cloud['colors'] = torch.from_numpy(np.asarray(cleaned.colors)).float().to(point_cloud['points'].device)

            info = f"Removed outliers using {method}\n"
            info += f"Original points: {len(pcd.points)}\n"
            info += f"Cleaned points: {len(cleaned.points)}\n"
            info += f"Removed points: {len(pcd.points) - len(cleaned.points)}\n"
            info += f"Removal percentage: {(1 - len(cleaned.points)/len(pcd.points))*100:.1f}%"

            return (new_point_cloud, info)

        except Exception as e:
            raise RuntimeError(f"Outlier removal failed: {e}")

class RDAWG3DPointCloudToMesh:
    """Reconstruct mesh from point cloud"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "point_cloud": ("POINT_CLOUD",),
                "method": (["poisson", "alpha_shape", "ball_pivoting"], {"default": "poisson"}),
                "depth": ("INT", {"default": 8, "min": 1, "max": 12}),
                "alpha": ("FLOAT", {"default": 0.03, "min": 0.001, "max": 0.5, "step": 0.001}),
                "radii": ("STRING", {"default": "0.005,0.01,0.02,0.04"}),
            }
        }

    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("mesh", "info")
    FUNCTION = "point_cloud_to_mesh"
    CATEGORY = "RDAWG 3D/Point Cloud"

    def point_cloud_to_mesh(self, point_cloud: dict, method: str, depth: int,
                           alpha: float, radii: str):
        if 'open3d_pcd' not in point_cloud:
            raise ValueError("Point cloud must have Open3D data")

        pcd = point_cloud['open3d_pcd']

        # Ensure normals are available
        if not pcd.has_normals():
            pcd.estimate_normals()
            pcd.orient_normals_consistent_tangent_plane(100)

        try:
            if method == "poisson":
                with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
                    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        pcd, depth=depth
                    )
                # Optional: Remove low density vertices
                vertices_to_remove = densities < np.quantile(densities, 0.01)
                mesh.remove_vertices_by_mask(vertices_to_remove)

            elif method == "alpha_shape":
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

            else:  # ball_pivoting
                radii_list = [float(r.strip()) for r in radii.split(',')]
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii_list)

            # Compute normals for the mesh
            mesh.compute_vertex_normals()

            # Create mesh data structure
            device = point_cloud['points'].device
            mesh_data = {
                'vertices': torch.from_numpy(np.asarray(mesh.vertices)).float().to(device),
                'faces': torch.from_numpy(np.asarray(mesh.triangles)).long().to(device),
                'open3d_mesh': mesh,
                'format': 'reconstructed_mesh'
            }

            if mesh.has_vertex_normals():
                mesh_data['vertex_normals'] = torch.from_numpy(np.asarray(mesh.vertex_normals)).float().to(device)

            info = f"Reconstructed mesh using {method}\n"
            info += f"Original points: {len(pcd.points)}\n"
            info += f"Mesh vertices: {len(mesh.vertices)}\n"
            info += f"Mesh faces: {len(mesh.triangles)}\n"

            if method == "poisson":
                info += f"Depth: {depth}"
            elif method == "alpha_shape":
                info += f"Alpha: {alpha}"
            else:
                info += f"Radii: {radii}"

            return (mesh_data, info)

        except Exception as e:
            raise RuntimeError(f"Mesh reconstruction failed: {e}")

class RDAWG3DTransformPointCloud:
    """Apply transformations to point cloud"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "point_cloud": ("POINT_CLOUD",),
                "scale": ("FLOAT", {"default": 1.0, "step": 0.01}),
                "rotation_x": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "rotation_y": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "rotation_z": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "translate_x": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "translate_y": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "translate_z": ("FLOAT", {"default": 0.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("POINT_CLOUD",)
    RETURN_NAMES = ("transformed_pcd",)
    FUNCTION = "transform_point_cloud"
    CATEGORY = "RDAWG 3D/Point Cloud"

    def transform_point_cloud(self, point_cloud: dict, scale: float, rotation_x: float,
                            rotation_y: float, rotation_z: float, translate_x: float,
                            translate_y: float, translate_z: float):

        if 'open3d_pcd' not in point_cloud:
            raise ValueError("Point cloud must have Open3D data")

        pcd = copy.deepcopy(point_cloud['open3d_pcd'])
        device = point_cloud['points'].device

        # Create transformation matrix
        # Scale
        if scale != 1.0:
            pcd.scale(scale, center=pcd.get_center())

        # Rotations (convert degrees to radians)
        if rotation_x != 0.0 or rotation_y != 0.0 or rotation_z != 0.0:
            # Create rotation matrices
            rx = np.radians(rotation_x)
            ry = np.radians(rotation_y)
            rz = np.radians(rotation_z)

            # Combined rotation matrix
            R = pcd.get_rotation_matrix_from_xyz((rx, ry, rz))
            pcd.rotate(R, center=pcd.get_center())

        # Translation
        if translate_x != 0.0 or translate_y != 0.0 or translate_z != 0.0:
            translation = np.array([translate_x, translate_y, translate_z])
            pcd.translate(translation)

        # Create new point cloud data
        new_point_cloud = point_cloud.copy()
        new_point_cloud['open3d_pcd'] = pcd
        new_point_cloud['points'] = torch.from_numpy(np.asarray(pcd.points)).float().to(device)

        if pcd.has_normals():
            # Transform normals also
            new_point_cloud['normals'] = torch.from_numpy(np.asarray(pcd.normals)).float().to(device)

        if pcd.has_colors():
            new_point_cloud['colors'] = torch.from_numpy(np.asarray(pcd.colors)).float().to(device)

        return (new_point_cloud,)

# Add to node mappings
NODE_CLASS_MAPPINGS = {
    "RDAWG3DMeshToPointCloud": RDAWG3DMeshToPointCloud,
    "RDAWG3DLoadPointCloud": RDAWG3DLoadPointCloud,
    "RDAWG3DDownsamplePointCloud": RDAWG3DDownsamplePointCloud,
    "RDAWG3DRemoveOutliers": RDAWG3DRemoveOutliers,
    "RDAWG3DPointCloudToMesh": RDAWG3DPointCloudToMesh,
    "RDAWG3DTransformPointCloud": RDAWG3DTransformPointCloud,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RDAWG3DMeshToPointCloud": "🔷 Mesh to Point Cloud (RDAWG)",
    "RDAWG3DLoadPointCloud": "🔷 Load Point Cloud (RDAWG)",
    "RDAWG3DDownsamplePointCloud": "🔷 Downsample Point Cloud (RDAWG)",
    "RDAWG3DRemoveOutliers": "🔷 Remove Outliers (RDAWG)",
    "RDAWG3DPointCloudToMesh": "🔷 Point Cloud to Mesh (RDAWG)",
    "RDAWG3DTransformPointCloud": "🔷 Transform Point Cloud (RDAWG)",
}