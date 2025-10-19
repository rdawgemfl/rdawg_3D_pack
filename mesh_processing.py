"""
Advanced Mesh Processing Nodes for RDAWG 3D Pack
Utilizes Open3D's advanced mesh processing capabilities
"""

import torch
import numpy as np
import open3d as o3d
from typing import Optional, Tuple, Any
import copy

class RDAWG3DSimplifyMesh:
    """Simplify mesh using Open3D's decimation algorithms"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "target_triangles": ("INT", {"default": 1000, "min": 100, "max": 100000}),
                "preserve_boundaries": ("BOOLEAN", {"default": True}),
                "method": (["quadric", "shortest_edge"], {"default": "quadric"}),
            }
        }

    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("simplified_mesh", "info")
    FUNCTION = "simplify_mesh"
    CATEGORY = "RDAWG 3D/Processing"

    def simplify_mesh(self, mesh: dict, target_triangles: int, preserve_boundaries: bool, method: str):
        if 'open3d_mesh' not in mesh:
            raise ValueError("Mesh must have Open3D mesh data")

        o3d_mesh = copy.deepcopy(mesh['open3d_mesh'])

        if len(o3d_mesh.triangles) <= target_triangles:
            return (mesh, f"Mesh already has {len(o3d_mesh.triangles)} triangles (≤ target)")

        try:
            if method == "quadric":
                # Quadric decimation
                simplified_mesh = o3d_mesh.simplify_quadric_decimation(
                    target_number_of_triangles=target_triangles,
                    preserve_boundary=preserve_boundaries
                )
            else:
                # Shortest edge decimation
                simplified_mesh = o3d_mesh.simplify_vertex_clustering(
                    voxel_size=o3d_mesh.get_max_bound() / (target_triangles ** 0.33) * 2
                )

            # Create new mesh data
            new_mesh = mesh.copy()
            new_mesh['open3d_mesh'] = simplified_mesh
            new_mesh['vertices'] = torch.from_numpy(np.asarray(simplified_mesh.vertices)).float().to(mesh['vertices'].device)
            new_mesh['faces'] = torch.from_numpy(np.asarray(simplified_mesh.triangles)).long().to(mesh['faces'].device) if mesh['faces'] is not None else None

            simplified_mesh.compute_vertex_normals()
            new_mesh['vertex_normals'] = torch.from_numpy(np.asarray(simplified_mesh.vertex_normals)).float().to(mesh['vertices'].device) if simplified_mesh.has_vertex_normals() else None

            info = f"Simplified mesh using {method}\n"
            info += f"Original triangles: {len(o3d_mesh.triangles)}\n"
            info += f"Simplified triangles: {len(simplified_mesh.triangles)}\n"
            info += f"Reduction: {(1 - len(simplified_mesh.triangles)/len(o3d_mesh.triangles))*100:.1f}%"

            return (new_mesh, info)

        except Exception as e:
            raise RuntimeError(f"Mesh simplification failed: {e}")

class RDAWG3DSubdivideMesh:
    """Subdivide mesh using different subdivision schemes"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "subdivisions": ("INT", {"default": 1, "min": 1, "max": 4}),
                "method": (["linear", "loop"], {"default": "loop"}),
            }
        }

    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("subdivided_mesh", "info")
    FUNCTION = "subdivide_mesh"
    CATEGORY = "RDAWG 3D/Processing"

    def subdivide_mesh(self, mesh: dict, subdivisions: int, method: str):
        if 'open3d_mesh' not in mesh:
            raise ValueError("Mesh must have Open3D mesh data")

        o3d_mesh = copy.deepcopy(mesh['open3d_mesh'])

        try:
            for _ in range(subdivisions):
                if method == "linear":
                    o3d_mesh = o3d_mesh.subdivide_midpoint()
                else:  # loop
                    o3d_mesh = o3d_mesh.subdivide_loop()

            # Create new mesh data
            new_mesh = mesh.copy()
            new_mesh['open3d_mesh'] = o3d_mesh
            new_mesh['vertices'] = torch.from_numpy(np.asarray(o3d_mesh.vertices)).float().to(mesh['vertices'].device)
            new_mesh['faces'] = torch.from_numpy(np.asarray(o3d_mesh.triangles)).long().to(mesh['faces'].device) if mesh['faces'] is not None else None

            o3d_mesh.compute_vertex_normals()
            new_mesh['vertex_normals'] = torch.from_numpy(np.asarray(o3d_mesh.vertex_normals)).float().to(mesh['vertices'].device) if o3d_mesh.has_vertex_normals() else None

            info = f"Subdivided mesh using {method}\n"
            info += f"Subdivisions: {subdivisions}\n"
            info += f"Original vertices: {len(mesh['vertices'])}\n"
            info += f"Subdivided vertices: {len(o3d_mesh.vertices)}\n"
            info += f"Original faces: {len(mesh['faces']) if mesh['faces'] is not None else 0}\n"
            info += f"Subdivided faces: {len(o3d_mesh.triangles)}"

            return (new_mesh, info)

        except Exception as e:
            raise RuntimeError(f"Mesh subdivision failed: {e}")

class RDAWG3DSmoothMesh:
    """Smooth mesh using various algorithms"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "iterations": ("INT", {"default": 1, "min": 1, "max": 50}),
                "lambda_factor": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 2.0, "step": 0.01}),
                "method": (["taubin", "laplacian"], {"default": "taubin"}),
            }
        }

    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("smoothed_mesh", "info")
    FUNCTION = "smooth_mesh"
    CATEGORY = "RDAWG 3D/Processing"

    def smooth_mesh(self, mesh: dict, iterations: int, lambda_factor: float, method: str):
        if 'open3d_mesh' not in mesh:
            raise ValueError("Mesh must have Open3D mesh data")

        o3d_mesh = copy.deepcopy(mesh['open3d_mesh'])

        try:
            if method == "taubin":
                # Taubin smoothing
                o3d_mesh.filter_smooth_taubin(number_of_iterations=iterations, lambda_filter=lambda_factor)
            else:
                # Laplacian smoothing
                o3d_mesh.filter_smooth_laplacian(number_of_iterations=iterations, lambda_filter=lambda_factor)

            # Create new mesh data
            new_mesh = mesh.copy()
            new_mesh['open3d_mesh'] = o3d_mesh
            new_mesh['vertices'] = torch.from_numpy(np.asarray(o3d_mesh.vertices)).float().to(mesh['vertices'].device)

            if o3d_mesh.has_vertex_normals():
                new_mesh['vertex_normals'] = torch.from_numpy(np.asarray(o3d_mesh.vertex_normals)).float().to(mesh['vertices'].device)

            info = f"Smoothed mesh using {method}\n"
            info += f"Iterations: {iterations}\n"
            info += f"Lambda factor: {lambda_factor}\n"
            info += f"Vertices: {len(o3d_mesh.vertices)}"

            return (new_mesh, info)

        except Exception as e:
            raise RuntimeError(f"Mesh smoothing failed: {e}")

class RDAWG3DRemeshUniform:
    """Remesh mesh to uniform edge lengths"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "target_edge_length": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 1.0, "step": 0.001}),
                "iterations": ("INT", {"default": 10, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("remeshed_mesh", "info")
    FUNCTION = "remesh_uniform"
    CATEGORY = "RDAWG 3D/Processing"

    def remesh_uniform(self, mesh: dict, target_edge_length: float, iterations: int):
        if 'open3d_mesh' not in mesh:
            raise ValueError("Mesh must have Open3D mesh data")

        o3d_mesh = copy.deepcopy(mesh['open3d_mesh'])

        try:
            # Create a simple uniform remeshing using voxel decimation and subdivision
            # This is a simplified approach - for production use, consider more advanced remeshing

            # First, estimate current edge length
            vertices = np.asarray(o3d_mesh.vertices)
            triangles = np.asarray(o3d_mesh.triangles)

            edge_lengths = []
            for triangle in triangles:
                for i in range(3):
                    v1_idx, v2_idx = triangle[i], triangle[(i+1)%3]
                    edge_length = np.linalg.norm(vertices[v1_idx] - vertices[v2_idx])
                    edge_lengths.append(edge_length)

            current_avg_edge = np.mean(edge_lengths) if edge_lengths else 0.01
            scale_factor = current_avg_edge / target_edge_length

            # Apply appropriate operation based on scale factor
            if scale_factor > 1.5:  # Need to subdivide
                subdivisions = min(3, int(np.log2(scale_factor)))
                for _ in range(subdivisions):
                    o3d_mesh = o3d_mesh.subdivide_loop()
            elif scale_factor < 0.7:  # Need to decimate
                target_triangles = int(len(o3d_mesh.triangles) * scale_factor ** 2)
                target_triangles = max(target_triangles, 100)  # Minimum triangles
                o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)

            # Create new mesh data
            new_mesh = mesh.copy()
            new_mesh['open3d_mesh'] = o3d_mesh
            new_mesh['vertices'] = torch.from_numpy(np.asarray(o3d_mesh.vertices)).float().to(mesh['vertices'].device)
            new_mesh['faces'] = torch.from_numpy(np.asarray(o3d_mesh.triangles)).long().to(mesh['faces'].device) if mesh['faces'] is not None else None

            o3d_mesh.compute_vertex_normals()
            new_mesh['vertex_normals'] = torch.from_numpy(np.asarray(o3d_mesh.vertex_normals)).float().to(mesh['vertices'].device) if o3d_mesh.has_vertex_normals() else None

            # Calculate new average edge length
            new_vertices = np.asarray(o3d_mesh.vertices)
            new_triangles = np.asarray(o3d_mesh.triangles)
            new_edge_lengths = []

            for triangle in new_triangles:
                for i in range(3):
                    v1_idx, v2_idx = triangle[i], triangle[(i+1)%3]
                    edge_length = np.linalg.norm(new_vertices[v1_idx] - new_vertices[v2_idx])
                    new_edge_lengths.append(edge_length)

            new_avg_edge = np.mean(new_edge_lengths) if new_edge_lengths else target_edge_length

            info = f"Uniform remeshing completed\n"
            info += f"Target edge length: {target_edge_length:.4f}\n"
            info += f"Original avg edge: {current_avg_edge:.4f}\n"
            info += f"New avg edge: {new_avg_edge:.4f}\n"
            info += f"Original triangles: {len(triangles)}\n"
            info += f"New triangles: {len(new_triangles)}"

            return (new_mesh, info)

        except Exception as e:
            raise RuntimeError(f"Mesh remeshing failed: {e}")

class RDAWG3DComputeCurvature:
    """Compute curvature information for mesh vertices"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "curvature_type": (["mean", "gaussian", "principal"], {"default": "mean"}),
                "radius": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MESH", "TENSOR", "STRING")
    RETURN_NAMES = ("mesh", "curvature", "info")
    FUNCTION = "compute_curvature"
    CATEGORY = "RDAWG 3D/Processing"

    def compute_curvature(self, mesh: dict, curvature_type: str, radius: float):
        if 'open3d_mesh' not in mesh:
            raise ValueError("Mesh must have Open3D mesh data")

        o3d_mesh = mesh['open3d_mesh']

        try:
            # Compute curvature using Open3D's estimate_normals and then curvature calculation
            # Open3D doesn't have built-in curvature, so we'll estimate it using normal differences

            if not o3d_mesh.has_vertex_normals():
                o3d_mesh.compute_vertex_normals()

            vertices = np.asarray(o3d_mesh.vertices)
            normals = np.asarray(o3d_mesh.vertex_normals)
            triangles = np.asarray(o3d_mesh.triangles)

            # Simple curvature estimation based on normal variations
            curvatures = np.zeros(len(vertices))

            if curvature_type == "mean":
                # Mean curvature estimation based on normal variation
                kdtree = o3d.geometry.KDTreeFlann(o3d_mesh)
                for i in range(len(vertices)):
                    _, idx, _ = kdtree.search_radius_vector_3d(o3d_mesh.vertices[i], radius)
                    if len(idx) > 1:
                        neighbor_normals = normals[idx[1:]]  # Exclude self
                        normal_diff = np.linalg.norm(neighbor_normals - normals[i], axis=1)
                        curvatures[i] = np.mean(normal_diff)

            elif curvature_type == "gaussian":
                # Gaussian curvature approximation
                for i, vertex in enumerate(vertices):
                    # Find triangles containing this vertex
                    vertex_triangles = [t for t in triangles if i in t]
                    if len(vertex_triangles) >= 3:
                        # Simple Gaussian curvature approximation
                        angle_sum = 0
                        for triangle in vertex_triangles:
                            # Get the other two vertices
                            other_vertices = [v for v in triangle if v != i]
                            if len(other_vertices) == 2:
                                v1, v2 = vertices[other_vertices[0]], vertices[other_vertices[1]]
                                # Compute angle at this vertex
                                edge1 = v1 - vertex
                                edge2 = v2 - vertex
                                cos_angle = np.dot(edge1, edge2) / (np.linalg.norm(edge1) * np.linalg.norm(edge2))
                                angle_sum += np.arccos(np.clip(cos_angle, -1, 1))

                        # Gaussian curvature approximation
                        curvatures[i] = (2 * np.pi - angle_sum) / len(vertex_triangles)

            else:  # principal
                # Principal curvature approximation (using shape operator)
                for i in range(len(vertices)):
                    kdtree = o3d.geometry.KDTreeFlann(o3d_mesh)
                    _, idx, _ = kdtree.search_radius_vector_3d(o3d_mesh.vertices[i], radius)
                    if len(idx) > 3:
                        local_vertices = vertices[idx] - vertices[i]
                        local_normals = normals[idx]

                        # Build covariance matrix of normals
                        cov_matrix = np.cov(local_normals.T)
                        eigenvalues = np.linalg.eigvals(cov_matrix)
                        curvatures[i] = np.max(np.abs(eigenvalues))

            # Normalize curvatures
            if np.max(curvatures) > np.min(curvatures):
                curvatures = (curvatures - np.min(curvatures)) / (np.max(curvatures) - np.min(curvatures))

            # Convert to tensor
            curvature_tensor = torch.from_numpy(curvatures).float().to(mesh['vertices'].device)

            info = f"Computed {curvature_type} curvature\n"
            info += f"Search radius: {radius}\n"
            info += f"Min curvature: {np.min(curvatures):.4f}\n"
            info += f"Max curvature: {np.max(curvatures):.4f}\n"
            info += f"Mean curvature: {np.mean(curvatures):.4f}"

            return (mesh, curvature_tensor, info)

        except Exception as e:
            raise RuntimeError(f"Curvature computation failed: {e}")

# Add to node mappings
NODE_CLASS_MAPPINGS = {
    "RDAWG3DSimplifyMesh": RDAWG3DSimplifyMesh,
    "RDAWG3DSubdivideMesh": RDAWG3DSubdivideMesh,
    "RDAWG3DSmoothMesh": RDAWG3DSmoothMesh,
    "RDAWG3DRemeshUniform": RDAWG3DRemeshUniform,
    "RDAWG3DComputeCurvature": RDAWG3DComputeCurvature,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RDAWG3DSimplifyMesh": "🔷 Simplify Mesh (RDAWG)",
    "RDAWG3DSubdivideMesh": "🔷 Subdivide Mesh (RDAWG)",
    "RDAWG3DSmoothMesh": "🔷 Smooth Mesh (RDAWG)",
    "RDAWG3DRemeshUniform": "🔷 Remesh Uniform (RDAWG)",
    "RDAWG3DComputeCurvature": "🔷 Compute Curvature (RDAWG)",
}