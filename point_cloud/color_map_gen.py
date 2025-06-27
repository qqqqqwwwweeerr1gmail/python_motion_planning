import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt


ply_file = r"/point_cloud/datas/total.ply"
ply_file = r"E:\GIT1\python_motion_planning\point_cloud\Depth2PC\workspace\0_pc_world_space.ply"
ply_file = r"E:\GIT1\python_motion_planning\point_cloud\Depth2PC\workspace\0_pc_camera_space.ply"
ply_data = PlyData.read(ply_file)
vertices = ply_data["vertex"]

# Extract coordinates (Y = height)
x = vertices["x"]
y = vertices["y"]  # Height
z = vertices["z"]

# Extract RGB color (assuming 8-bit: 0-255)
r = vertices["red"] / 255.0    # Normalize to [0, 1]
g = vertices["green"] / 255.0
b = vertices["blue"] / 255.0
colors = np.vstack((r, g, b)).T  # Shape: (N, 3)






# For X-Y view (vertical slice along Z-axis)
x_plot = x
y_plot = y

# For Z-Y view (vertical slice along X-axis)
# x_plot = z
# y_plot = y




plt.figure(figsize=(6, 10))  # Taller than wide (vertical)
plt.scatter(
    x_plot, y_plot,
    c=colors,  # Use RGB colors from PLY
    s=0.1,     # Adjust point size
    marker="."  # Small dots
)
plt.xlabel("X (or Z) Coordinate")
plt.ylabel("Height (Y)")
plt.title("Vertical Projection (Colored by PLY Data)")
plt.gca().set_aspect("equal")  # Avoid axis distortion
plt.savefig("vertical_view.png", dpi=300, bbox_inches="tight")
plt.close()



import open3d as o3d

# Create point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.vstack((x, y, z)).T)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Set camera to view Y-up (vertical)
vis = o3d.visualization.Visualizer()
vis.create_window(width=600, height=1000)  # Vertical window
vis.add_geometry(pcd)

# Adjust view to look at X-Y or Z-Y plane
ctrl = vis.get_view_control()
ctrl.set_up([0, 0, 1])  # Z-axis is up (adjust if needed)
ctrl.set_front([0, -1, 0])  # Look at Y-axis
ctrl.set_lookat([0, 0, 0])

# Save screenshot
vis.capture_screen_image("vertical_view_op3d.png")
vis.destroy_window()







