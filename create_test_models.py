#!/usr/bin/env python3
"""
Create test 3D models for RDAWG 3D Pack testing
"""

import os
import numpy as np

def create_cube_obj(filename="cube.obj", size=1.0):
    """Create a simple cube OBJ file"""
    vertices = [
        [-size, -size, -size], [size, -size, -size], [size, size, -size], [-size, size, -size],
        [-size, -size, size], [size, -size, size], [size, size, size], [-size, size, size]
    ]

    faces = [
        [1, 2, 3], [1, 3, 4],  # bottom
        [5, 8, 7], [5, 7, 6],  # top
        [1, 5, 6], [1, 6, 2],  # front
        [2, 6, 7], [2, 7, 3],  # right
        [3, 7, 8], [3, 8, 4],  # back
        [4, 8, 5], [4, 5, 1]   # left
    ]

    with open(filename, 'w') as f:
        f.write("# Test Cube\n")
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    print(f"Created {filename}")

def create_sphere_obj(filename="sphere.obj", radius=1.0, segments=16):
    """Create a sphere OBJ file"""
    vertices = []
    faces = []

    # Generate vertices
    for i in range(segments + 1):
        lat = np.pi * i / segments - np.pi / 2
        for j in range(segments):
            lon = 2 * np.pi * j / segments
            x = radius * np.cos(lat) * np.cos(lon)
            y = radius * np.sin(lat)
            z = radius * np.cos(lat) * np.sin(lon)
            vertices.append([x, y, z])

    # Generate faces
    for i in range(segments):
        for j in range(segments):
            current = i * segments + j
            next = current + segments
            next_j = current + 1
            next_next = next + 1

            if j < segments - 1:
                faces.append([current + 1, next + 1, next_j + 1])
                faces.append([current + 1, next_j + 1, next_j + 1])

    with open(filename, 'w') as f:
        f.write("# Test Sphere\n")
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces[:len(faces)//2]:  # Limit faces for simplicity
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    print(f"Created {filename}")

def create_pyramid_obj(filename="pyramid.obj", size=1.0):
    """Create a pyramid OBJ file"""
    vertices = [
        [-size, 0, -size], [size, 0, -size], [size, 0, size], [-size, 0, size],  # base
        [0, size*2, 0]  # apex
    ]

    faces = [
        [1, 2, 3], [1, 3, 4],  # base
        [1, 5, 2],  # front
        [2, 5, 3],  # right
        [3, 5, 4],  # back
        [4, 5, 1]   # left
    ]

    with open(filename, 'w') as f:
        f.write("# Test Pyramid\n")
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    print(f"Created {filename}")

def main():
    """Create all test models"""
    print("Creating test 3D models for RDAWG 3D Pack...")

    create_cube_obj("test_cube.obj", 1.0)
    create_sphere_obj("test_sphere.obj", 1.0, 12)
    create_pyramid_obj("test_pyramid.obj", 1.0)

    print("\nTest models created successfully!")
    print("You can now use these files to test RDAWG 3D Pack in ComfyUI:")
    print("- test_cube.obj")
    print("- test_sphere.obj")
    print("- test_pyramid.obj")

if __name__ == "__main__":
    main()