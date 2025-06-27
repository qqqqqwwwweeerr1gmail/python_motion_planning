import trimesh

# Load the PLY file
mesh = trimesh.load(rf'E:\GIT1\python_motion_planning\point_cloud\total_world_space_pc.ply')

# # Check if the mesh loaded correctly
# if isinstance(mesh, trimesh.Trimesh):
#     print("Mesh loaded successfully!")
#     print(f"Vertices: {mesh.vertices.shape}")  # (n, 3) array of vertices
#     print(f"Faces: {mesh.faces.shape}")        # (m, 3) array of triangles
# else:
#     print("Failed to load mesh.")


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt



import trimesh
from pathlib import Path

# Define the path to your PLY file
# Replace 'path/to/your/file.ply' with the actual path to your PLY file.
# For example, if it's in the same directory as your script:
ply_file_path = Path(rf'E:\GIT1\python_motion_planning\point_cloud\total_world_space_pc(1).ply')

# Or if it's an absolute path:
# ply_file_path = Path("E:/GIT1/python_motion_planning/data/my_mesh.ply")

try:
    # Load the PLY file
    # trimesh.load will automatically detect the file type from the extension.
    mesh = trimesh.load(ply_file_path)

    print(f"Successfully loaded mesh from: {ply_file_path}")
    print(f"Number of vertices: {len(mesh.vertices)}")
    # print(f"Number of faces: {len(mesh.faces)}")

    # You can access mesh properties:
    # mesh.vertices: A (n, 3) numpy array of vertex coordinates
    # mesh.faces: A (m, 3) numpy array of face indices (triangles)
    # mesh.visual.vertex_colors: If the PLY has vertex colors
    # mesh.visual.face_colors: If the PLY has face colors
    # mesh.is_watertight: Check if the mesh is watertight
    # mesh.volume: Calculate the volume if watertight
    # etc.

    a = mesh.vertices
    print(f"Vertices a: {a}")
    x = [i[0] for i in a]
    y = [i[1] for i in a]
    z = [i[2] for i in a]
    print(f"Vertices x: {x}")
    print(f"Vertices y: {y}")
    print(f"Vertices z: {z}")

    unique_values, counts = np.unique(x, return_counts=True)

    # Plot
    plt.plot(unique_values, counts, marker='o', linestyle='-', color='blue')
    plt.xlabel("Value (Sorted)")
    plt.ylabel("Frequency")
    plt.title("Value Frequency Distribution (Line Plot)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    unique_values, counts = np.unique(y, return_counts=True)

    # Plot
    plt.plot(unique_values, counts, marker='o', linestyle='-', color='blue')
    plt.xlabel("Value (Sorted)")
    plt.ylabel("Frequency")
    plt.title("Value Frequency Distribution (Line Plot)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


    unique_values, counts = np.unique(z, return_counts=True)

    # Plot
    plt.plot(unique_values, counts, marker='o', linestyle='-', color='blue')
    plt.xlabel("Value (Sorted)")
    plt.ylabel("Frequency")
    plt.title("Value Frequency Distribution (Line Plot)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    # Example: Check if the mesh has vertex colors
    if mesh.visual.vertex_colors is not None:
        print(f"Mesh has vertex colors. Shape: {mesh.visual.vertex_colors.shape}")
    else:
        print("Mesh does not have vertex colors.")

    # Example: Show the mesh (requires pyglet or other visualization dependencies)
    # mesh.show()

except FileNotFoundError:
    print(f"Error: The file '{ply_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred while loading the PLY file: {e}")









