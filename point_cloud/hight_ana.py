
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData
from matplotlib.colors import ListedColormap

from plyfile import PlyData
import matplotlib.pyplot as plt




ply_file = r"/point_cloud/datas/total.ply"  # Replace with your PLY file
ply_file = "your_platform.ply"  # Replace with your PLY file

dpi = 200

height_threshold = 1



ply_data = PlyData.read(ply_file)
vertices = ply_data['vertex']
x = vertices['x']
y = vertices['y']
z = vertices['z']
z = [-i for i in z]

min_x, max_x = np.min(x), np.max(x)
min_z, max_z = np.min(z), np.max(z)
min_y, max_y = np.min(y), np.max(y)

grid_scale_x = int((max_x - min_x) / dpi)
grid_scale_z = int((max_z - min_z) / dpi)

x_bins = np.arange(min_x, max_x + grid_scale_x, grid_scale_x)
z_bins = np.arange(min_z, max_z + grid_scale_z, grid_scale_z)

grid_max = np.full((len(x_bins) - 1, len(z_bins) - 1), -np.inf)
grid_min = np.full((len(x_bins) - 1, len(z_bins) - 1), np.inf)

height_var = grid_max - grid_min

walkable = np.ones_like(height_var, dtype=int)
walkable[np.isinf(height_var)] = 0  # No data cells
walkable[height_var > height_threshold] = 0  # High variation cells

return walkable, (min_x, min_z), grid_size








plt.figure(figsize=(10, 8))
scatter = plt.scatter(x, z, c=y, cmap='viridis', s=1)  # 's' adjusts point size
plt.colorbar(scatter, label='Height (Y)')
plt.xlabel('X Coordinate')
plt.ylabel('Z Coordinate')
plt.title('Top-Down View (X-Z Plane) with Height Coloring')
plt.grid(True)











