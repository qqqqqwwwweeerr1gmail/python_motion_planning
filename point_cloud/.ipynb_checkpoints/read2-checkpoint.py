import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh

# Load PLY file
mesh = trimesh.load(rf'E:\GIT1\python_motion_planning\point_cloud\total_world_space_pc(1).ply')

# Extract vertices (assuming it's a point cloud)
points = mesh.vertices  # Shape: (N, 3)

points = points[::10]

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(
    points[:, 0],  # X-axis
    points[:, 1],  # Y-axis
    points[:, 2],  # Z-axis
    s=1,           # Point size
    c=points[:, 2], # Color by height (Z-axis)
    cmap='viridis', # Color map
    alpha=0.5       # Transparency
)

# Labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Point Cloud (Matplotlib)')

plt.show()