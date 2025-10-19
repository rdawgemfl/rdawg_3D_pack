"""
Mesh Analysis and Utility Nodes for RDAWG 3D Pack
Provides comprehensive mesh analysis and utility functions
"""

import torch
import numpy as np
import open3d as o3d
from typing import Optional, Tuple, Any, Dict
import copy

class RDAWG3DAnalyzeMesh:
    """Comprehensive mesh analysis providing various metrics"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "compute_volume": ("BOOLEAN", {"default": True}),
                "compute_surface_area": ("BOOLEAN", {"default": True}),
                "compute_quality_metrics": ("BOOLEAN", {"default": True}),
                "compute_topology": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("basic_info", "geometry_metrics", "quality_metrics")
    FUNCTION = "analyze_mesh"
    CATEGORY = "RDAWG 3D/Analysis"

    def analyze_mesh(self, mesh: dict, compute_volume: bool, compute_surface_area: bool,
                    compute_quality_metrics: bool, compute_topology: bool):
        if 'open3d_mesh' not in mesh:
            raise ValueError("Mesh must have Open3D mesh data")

        o3d_mesh = mesh['open3d_mesh']

        # Basic information
        basic_info = "=== MESH BASIC INFO ===\n"
        basic_info += f"Vertices: {len(o3d_mesh.vertices)}\n"
        basic_info += f"Triangles: {len(o3d_mesh.triangles)}\n"
        basic_info += f"Has Vertex Normals: {o3d_mesh.has_vertex_normals()}\n"
        basic_info += f"Has Triangle Normals: {o3d_mesh.has_triangle_normals()}\n"

        if o3d_mesh.has_vertex_colors():
            basic_info += f"Has Vertex Colors: Yes\n"
        if hasattr(o3d_mesh, 'has_texture_uvs') and o3d_mesh.has_texture_uvs():
            basic_info += f"Has UV Coordinates: Yes\n"

        # Bounding box
        bbox = o3d_mesh.get_axis_aligned_bounding_box()
        basic_info += f"Bounding Box Size: {bbox.get_extent().tolist()}\n"
        basic_info += f"Center: {bbox.get_center().tolist()}\n"

        # Geometry metrics
        geometry_metrics = "=== GEOMETRY METRICS ===\n"

        if compute_surface_area:
            try:
                surface_area = o3d_mesh.get_surface_area()
                geometry_metrics += f"Surface Area: {surface_area:.6f}\n"
            except:
                geometry_metrics += "Surface Area: Failed to compute\n"

        if compute_volume:
            try:
                if o3d_mesh.is_watertight():
                    volume = o3d_mesh.get_volume()
                    geometry_metrics += f"Volume: {volume:.6f}\n"
                else:
                    geometry_metrics += "Volume: Mesh is not watertight\n"
            except:
                geometry_metrics += "Volume: Failed to compute\n"

        # Edge statistics
        vertices = np.asarray(o3d_mesh.vertices)
        triangles = np.asarray(o3d_mesh.triangles)

        if len(triangles) > 0:
            edges = set()
            for triangle in triangles:
                for i in range(3):
                    edge = tuple(sorted([triangle[i], triangle[(i+1)%3]]))
                    edges.add(edge)

            geometry_metrics += f"Unique Edges: {len(edges)}\n"

            # Edge length statistics
            edge_lengths = []
            for edge in edges:
                v1, v2 = vertices[edge[0]], vertices[edge[1]]
                edge_lengths.append(np.linalg.norm(v1 - v2))

            if edge_lengths:
                geometry_metrics += f"Edge Length - Min: {min(edge_lengths):.6f}\n"
                geometry_metrics += f"Edge Length - Max: {max(edge_lengths):.6f}\n"
                geometry_metrics += f"Edge Length - Mean: {np.mean(edge_lengths):.6f}\n"
                geometry_metrics += f"Edge Length - Std: {np.std(edge_lengths):.6f}\n"

        # Quality metrics
        quality_metrics = "=== QUALITY METRICS ===\n"

        if compute_quality_metrics:
            # Triangle quality metrics
            if len(triangles) > 0:
                areas = []
                aspect_ratios = []

                for triangle in triangles:
                    v0, v1, v2 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]

                    # Triangle area
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
                    areas.append(area)

                    # Aspect ratio (longest edge / shortest altitude)
                    edge_lengths = [
                        np.linalg.norm(v1 - v0),
                        np.linalg.norm(v2 - v1),
                        np.linalg.norm(v0 - v2)
                    ]
                    longest_edge = max(edge_lengths)
                    shortest_altitude = (2 * area) / longest_edge if area > 0 else 0
                    aspect_ratio = longest_edge / shortest_altitude if shortest_altitude > 0 else float('inf')
                    aspect_ratios.append(aspect_ratio)

                quality_metrics += f"Triangle Areas - Min: {min(areas):.6f}\n"
                quality_metrics += f"Triangle Areas - Max: {max(areas):.6f}\n"
                quality_metrics += f"Triangle Areas - Mean: {np.mean(areas):.6f}\n"
                quality_metrics += f"Aspect Ratios - Min: {min(aspect_ratios):.3f}\n"
                quality_metrics += f"Aspect Ratios - Max: {max(aspect_ratios):.3f}\n"
                quality_metrics += f"Aspect Ratios - Mean: {np.mean(aspect_ratios):.3f}\n"

                # Triangle count by quality
                good_triangles = sum(1 for ar in aspect_ratios if ar < 3.0)
                ok_triangles = sum(1 for ar in aspect_ratios if 3.0 <= ar < 10.0)
                bad_triangles = sum(1 for ar in aspect_ratios if ar >= 10.0)

                quality_metrics += f"Good Triangles (AR<3): {good_triangles} ({100*good_triangles/len(aspect_ratios):.1f}%)\n"
                quality_metrics += f"OK Triangles (3≤AR<10): {ok_triangles} ({100*ok_triangles/len(aspect_ratios):.1f}%)\n"
                quality_metrics += f"Bad Triangles (AR≥10): {bad_triangles} ({100*bad_triangles/len(aspect_ratios):.1f}%)\n"

        if compute_topology:
            # Topology analysis
            quality_metrics += "\n=== TOPOLOGY ===\n"
            quality_metrics += f"Is Watertight: {o3d_mesh.is_watertight()}\n"
            quality_metrics += f"Is Edge Manifold: {o3d_mesh.is_edge_manifold()}\n"
            quality_metrics += f"Is Vertex Manifold: {o3d_mesh.is_vertex_manifold()}\n"

            # Genus calculation (for closed manifolds)
            if o3d_mesh.is_watertight():
                # Euler characteristic: V - E + F = 2 - 2g (where g is genus)
                V = len(vertices)
                F = len(triangles)
                E = len(edges) if 'edges' in locals() else 0
                euler = V - E + F
                genus = (2 - euler) // 2 if euler <= 2 else 0
                quality_metrics += f"Euler Characteristic: {euler}\n"
                quality_metrics += f"Estimated Genus: {genus}\n"

        return (basic_info, geometry_metrics, quality_metrics)

class RDAWG3DComputeDistance:
    """Compute distances between meshes or within a mesh"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_a": ("MESH",),
                "distance_type": (["self_hausdorff", "mesh_to_mesh", "closest_points"], {"default": "self_hausdorff"}),
                "mesh_b": ("MESH",),
            }
        }

    RETURN_TYPES = ("STRING", "TENSOR")
    RETURN_NAMES = ("distance_info", "distance_values")
    FUNCTION = "compute_distance"
    CATEGORY = "RDAWG 3D/Analysis"

    def compute_distance(self, mesh_a: dict, distance_type: str, mesh_b: dict = None):
        if 'open3d_mesh' not in mesh_a:
            raise ValueError("Mesh A must have Open3D mesh data")

        o3d_mesh_a = mesh_a['open3d_mesh']

        try:
            if distance_type == "self_hausdorff":
                # Self Hausdorff distance
                distances = o3d_mesh_a.compute_self_intersection()
                info = "=== SELF-INTERSECTION ANALYSIS ===\n"
                if len(distances) > 0:
                    info += f"Self-intersections found: {len(distances)}\n"
                    info += f"Average intersection distance: {np.mean(distances):.6f}\n"
                else:
                    info += "No self-intersections detected\n"

                distance_tensor = torch.from_numpy(distances).float().to(mesh_a['vertices'].device) if len(distances) > 0 else torch.tensor([], device=mesh_a['vertices'].device)

            elif distance_type == "mesh_to_mesh":
                if mesh_b is None or 'open3d_mesh' not in mesh_b:
                    raise ValueError("Mesh B required for mesh-to-mesh distance")

                o3d_mesh_b = mesh_b['open3d_mesh']

                # Compute Hausdorff distance
                hausdorff_distance = o3d_mesh_a.compute_point_cloud_distance(o3d_mesh_b.vertices)

                info = "=== MESH-TO-MESH DISTANCE ===\n"
                info += f"Max Hausdorff Distance: {max(hausdorff_distance):.6f}\n"
                info += f"Mean Hausdorff Distance: {np.mean(hausdorff_distance):.6f}\n"
                info += f"Min Hausdorff Distance: {min(hausdorff_distance):.6f}\n"

                distance_tensor = torch.from_numpy(hausdorff_distance).float().to(mesh_a['vertices'].device)

            else:  # closest_points
                if mesh_b is None or 'open3d_mesh' not in mesh_b:
                    raise ValueError("Mesh B required for closest points distance")

                o3d_mesh_b = mesh_b['open3d_mesh']

                # Find closest points
                distances = o3d_mesh_a.compute_point_cloud_distance(o3d_mesh_b.vertices)

                info = "=== CLOSEST POINTS DISTANCE ===\n"
                info += f"Points in Mesh A: {len(o3d_mesh_a.vertices)}\n"
                info += f"Points in Mesh B: {len(o3d_mesh_b.vertices)}\n"
                info += f"Mean closest distance: {np.mean(distances):.6f}\n"
                info += f"Max closest distance: {max(distances):.6f}\n"

                distance_tensor = torch.from_numpy(distances).float().to(mesh_a['vertices'].device)

            return (info, distance_tensor)

        except Exception as e:
            raise RuntimeError(f"Distance computation failed: {e}")

class RDAWG3DBoundingBox:
    """Compute and manipulate bounding boxes"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "box_type": (["axis_aligned", "oriented"], {"default": "axis_aligned"}),
                "create_visualization": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "MESH", "TENSOR", "TENSOR")
    RETURN_NAMES = ("box_info", "box_mesh", "min_bounds", "max_bounds")
    FUNCTION = "compute_bounding_box"
    CATEGORY = "RDAWG 3D/Analysis"

    def compute_bounding_box(self, mesh: dict, box_type: str, create_visualization: bool):
        if 'open3d_mesh' not in mesh:
            raise ValueError("Mesh must have Open3D mesh data")

        o3d_mesh = mesh['open3d_mesh']

        try:
            if box_type == "axis_aligned":
                bbox = o3d_mesh.get_axis_aligned_bounding_box()
            else:
                bbox = o3d_mesh.get_oriented_bounding_box()

            # Bounding box information
            info = f"=== {box_type.upper()} BOUNDING BOX ===\n"
            info += f"Center: {bbox.get_center().tolist()}\n"
            info += f"Extent: {bbox.get_extent().tolist()}\n"
            info += f"Volume: {bbox.volume():.6f}\n"

            if box_type == "oriented":
                info += f"Orientation Matrix:\n"
                R = bbox.R
                for row in R:
                    info += f"  [{row[0]:.3f}, {row[1]:.3f}, {row[2]:.3f}]\n"

            # Create visualization mesh if requested
            box_mesh = None
            if create_visualization:
                box_mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(bbox)
                box_mesh.compute_vertex_normals()

                # Convert to RDAWG mesh format
                box_mesh_data = {
                    'vertices': torch.from_numpy(np.asarray(box_mesh.vertices)).float().to(mesh['vertices'].device),
                    'faces': torch.from_numpy(np.asarray(box_mesh.triangles)).long().to(mesh['vertices'].device),
                    'open3d_mesh': box_mesh,
                    'format': 'bounding_box'
                }

                if box_mesh.has_vertex_normals():
                    box_mesh_data['vertex_normals'] = torch.from_numpy(np.asarray(box_mesh.vertex_normals)).float().to(mesh['vertices'].device)

            # Min and max bounds
            min_bounds = torch.from_numpy(bbox.get_min_bound()).float().to(mesh['vertices'].device)
            max_bounds = torch.from_numpy(bbox.get_max_bound()).float().to(mesh['vertices'].device)

            return (info, box_mesh_data if create_visualization else None, min_bounds, max_bounds)

        except Exception as e:
            raise RuntimeError(f"Bounding box computation failed: {e}")

class RDAWG3DExtractFeatures:
    """Extract geometric features from mesh"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "feature_type": (["vertex_curvature", "edge_angles", "face_areas", "vertex_valence"], {"default": "vertex_curvature"}),
                "radius": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING", "TENSOR")
    RETURN_NAMES = ("feature_info", "features")
    FUNCTION = "extract_features"
    CATEGORY = "RDAWG 3D/Analysis"

    def extract_features(self, mesh: dict, feature_type: str, radius: float):
        if 'open3d_mesh' not in mesh:
            raise ValueError("Mesh must have Open3D mesh data")

        o3d_mesh = mesh['open3d_mesh']

        try:
            if feature_type == "vertex_curvature":
                # Estimate vertex curvature
                if not o3d_mesh.has_vertex_normals():
                    o3d_mesh.compute_vertex_normals()

                vertices = np.asarray(o3d_mesh.vertices)
                normals = np.asarray(o3d_mesh.vertex_normals)
                triangles = np.asarray(o3d_mesh.triangles)

                # Simple curvature estimation
                curvatures = np.zeros(len(vertices))
                kdtree = o3d.geometry.KDTreeFlann(o3d_mesh)

                for i in range(len(vertices)):
                    _, idx, _ = kdtree.search_radius_vector_3d(o3d_mesh.vertices[i], radius)
                    if len(idx) > 3:
                        local_vertices = vertices[idx] - vertices[i]
                        local_normals = normals[idx]
                        cov_matrix = np.cov(local_normals.T)
                        eigenvalues = np.linalg.eigvals(cov_matrix)
                        curvatures[i] = np.max(np.abs(eigenvalues))

                features = torch.from_numpy(curvatures).float().to(mesh['vertices'].device)

                info = "=== VERTEX CURVATURE ===\n"
                info += f"Search radius: {radius}\n"
                info += f"Min curvature: {np.min(curvatures):.6f}\n"
                info += f"Max curvature: {np.max(curvatures):.6f}\n"
                info += f"Mean curvature: {np.mean(curvatures):.6f}\n"

            elif feature_type == "edge_angles":
                # Compute edge angles (dihedral angles)
                triangles = np.asarray(o3d_mesh.triangles)
                vertices = np.asarray(o3d_mesh.vertices)

                # Build edge adjacency
                edge_to_triangles = {}
                for tri_idx, triangle in enumerate(triangles):
                    for i in range(3):
                        edge = tuple(sorted([triangle[i], triangle[(i+1)%3]]))
                        if edge not in edge_to_triangles:
                            edge_to_triangles[edge] = []
                        edge_to_triangles[edge].append(tri_idx)

                # Compute dihedral angles for interior edges
                angles = []
                for edge, tri_indices in edge_to_triangles.items():
                    if len(tri_indices) == 2:  # Interior edge
                        # Compute dihedral angle
                        tri1, tri2 = triangles[tri_indices[0]], triangles[tri_indices[1]]

                        # Compute face normals
                        v1, v2, v3 = vertices[tri1[0]], vertices[tri1[1]], vertices[tri1[2]]
                        normal1 = np.cross(v2 - v1, v3 - v1)
                        normal1 = normal1 / np.linalg.norm(normal1)

                        v1, v2, v3 = vertices[tri2[0]], vertices[tri2[1]], vertices[tri2[2]]
                        normal2 = np.cross(v2 - v1, v3 - v1)
                        normal2 = normal2 / np.linalg.norm(normal2)

                        # Dihedral angle
                        cos_angle = np.dot(normal1, normal2)
                        angle = np.arccos(np.clip(cos_angle, -1, 1))
                        angles.append(angle)

                features = torch.from_numpy(np.array(angles)).float().to(mesh['vertices'].device)

                info = "=== EDGE ANGLES (DIHEDRAL) ===\n"
                if angles:
                    info += f"Interior edges: {len(angles)}\n"
                    info += f"Min angle (degrees): {np.degrees(min(angles)):.2f}\n"
                    info += f"Max angle (degrees): {np.degrees(max(angles)):.2f}\n"
                    info += f"Mean angle (degrees): {np.degrees(np.mean(angles)):.2f}\n"

            elif feature_type == "face_areas":
                # Compute face areas
                triangles = np.asarray(o3d_mesh.triangles)
                vertices = np.asarray(o3d_mesh.vertices)

                areas = []
                for triangle in triangles:
                    v0, v1, v2 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
                    areas.append(area)

                features = torch.from_numpy(np.array(areas)).float().to(mesh['vertices'].device)

                info = "=== FACE AREAS ===\n"
                info += f"Number of faces: {len(areas)}\n"
                info += f"Min area: {min(areas):.6f}\n"
                info += f"Max area: {max(areas):.6f}\n"
                info += f"Mean area: {np.mean(areas):.6f}\n"
                info += f"Total area: {sum(areas):.6f}\n"

            else:  # vertex_valence
                # Compute vertex valence (number of connected edges)
                triangles = np.asarray(o3d_mesh.triangles)
                valences = {}

                for triangle in triangles:
                    for vertex_idx in triangle:
                        if vertex_idx not in valences:
                            valences[vertex_idx] = 0
                        valences[vertex_idx] += 1

                # Create array in vertex order
                vertex_count = len(o3d_mesh.vertices)
                valence_array = np.array([valences.get(i, 0) for i in range(vertex_count)])

                features = torch.from_numpy(valence_array).float().to(mesh['vertices'].device)

                info = "=== VERTEX VALENCE ===\n"
                info += f"Min valence: {min(valence_array)}\n"
                info += f"Max valence: {max(valence_array)}\n"
                info += f"Mean valence: {np.mean(valence_array):.2f}\n"

                # Count vertices by valence
                valence_counts = {}
                for valence in valence_array:
                    valence_counts[valence] = valence_counts.get(valence, 0) + 1

                info += "Valence distribution:\n"
                for valence, count in sorted(valence_counts.items()):
                    info += f"  Valence {valence}: {count} vertices\n"

            return (info, features)

        except Exception as e:
            raise RuntimeError(f"Feature extraction failed: {e}")

# Add to node mappings
NODE_CLASS_MAPPINGS = {
    "RDAWG3DAnalyzeMesh": RDAWG3DAnalyzeMesh,
    "RDAWG3DComputeDistance": RDAWG3DComputeDistance,
    "RDAWG3DBoundingBox": RDAWG3DBoundingBox,
    "RDAWG3DExtractFeatures": RDAWG3DExtractFeatures,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RDAWG3DAnalyzeMesh": "🔷 Analyze Mesh (RDAWG)",
    "RDAWG3DComputeDistance": "🔷 Compute Distance (RDAWG)",
    "RDAWG3DBoundingBox": "🔷 Bounding Box (RDAWG)",
    "RDAWG3DExtractFeatures": "🔷 Extract Features (RDAWG)",
}