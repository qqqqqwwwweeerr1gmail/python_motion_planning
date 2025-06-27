import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData
from matplotlib.colors import ListedColormap

import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt

ply_file = r"/point_cloud/datas/total.ply"  # Replace with your PLY file
ply_file = r"E:\GIT1\python_motion_planning\point_cloud\Depth2PC\workspace\0_pc_world_space.ply"  # Replace with your PLY file
ply_file = "your_platform.ply"  # Replace with your PLY file
ply_file = r"E:\GIT1\python_motion_planning\point_cloud\Depth2PC\workspace\0_pc_camera_space.ply"  # Replace with your PLY file

ply_data = PlyData.read(ply_file)

# Extract vertices (assuming the PLY contains 'vertex' element)
vertices = ply_data['vertex']

# Get x, y (height), z coordinates
x = vertices['x']
y = vertices['y']  # Height (Y-axis is typically "up" in 3D models)
z = vertices['z']
plt.figure(figsize=(10, 8))
scatter = plt.scatter(x, z, c=y, cmap='viridis', s=1)  # 's' adjusts point size
plt.colorbar(scatter, label='Height (Y)')
plt.xlabel('X Coordinate')
plt.ylabel('Z Coordinate')
plt.title('Top-Down View (X-Z Plane) with Height Coloring')
plt.grid(True)
plt.show()