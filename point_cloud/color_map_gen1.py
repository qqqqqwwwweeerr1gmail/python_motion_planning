import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# --- Configuration ---
ply_file = r"E:\GIT1\python_motion_planning\point_cloud\Depth2PC\workspace\0_pc_world_space.ply"  # Replace with your PLY file
ply_file = "your_platform.ply"  # Replace with your PLY file
ply_file = r"E:\GIT1\python_motion_planning\point_cloud\Depth2PC\workspace\0_pc_camera_space.ply"  # Replace with your PLY file
ply_file = r"/point_cloud/datas/total.ply"  # Replace with your PLY file

ply_file_path = ply_file

output_scan_png_path = "vertical_scan_graph.png"    # Output PNG filename

# --- 1. Read the PLY file ---
try:
    pcd = o3d.io.read_point_cloud(ply_file_path)
    if not pcd.has_points():
        print(f"Error: No points found in {ply_file_path}")
        exit()
    if not pcd.has_colors():
        print(f"Error: PLY file '{ply_file_path}' does not contain color information. Cannot generate colored scan graph.")
        exit()

except Exception as e:
    print(f"Error reading PLY file: {e}")
    print(f"Please ensure '{ply_file_path}' exists and is a valid PLY file with colors.")
    exit()

# Get the points and colors as NumPy arrays
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors) # Colors are R, G, B values between 0.0 and 1.0

# Separate coordinates
x = points[:, 0]
y = points[:, 1] # This is your 'height' coordinate
z = points[:, 2]

# --- 2. Sort points for "scanning" effect ---
# Choose which horizontal dimension to sort by. X is usually like a left-to-right scan.
# We'll create an index array that sorts points based on their X-coordinate.
sort_indices = np.argsort(x)

x_sorted = x[sort_indices]
y_sorted = y[sort_indices]
z_sorted = z[sort_indices] # (Optional, but good to keep consistent)
colors_sorted = colors[sort_indices]

# --- 3. Create the "Vertical Scan Graph" using Matplotlib ---
# We'll plot the sorted X values on the horizontal axis and Y (height) on the vertical axis.
# Each point will be colored with its original color.

fig, ax = plt.subplots(figsize=(15, 8)) # Adjust figure size for a wider scan view

# Use a scatter plot with very small marker size (s=1 or s=0.5) to simulate pixels
# The 'c' argument directly takes the (N, 3) array of RGB colors.
scatter = ax.scatter(x_sorted, y_sorted, c=colors_sorted, s=1, alpha=1.0) # s=1 for pixel-like dots

ax.set_xlabel('X Coordinate (Scan Progression)')
ax.set_ylabel('Height (Y Coordinate)')
ax.set_title('Vertical Scan Graph: Color by Height (Y vs X)')
ax.grid(True, linestyle='--', alpha=0.6)

# Set tight limits to make sure all data is visible
ax.set_xlim(x_sorted.min(), x_sorted.max())
ax.set_ylim(y_sorted.min(), y_sorted.max())

# Ensure the aspect ratio doesn't distort the height too much, or make it 'auto'
ax.set_aspect('auto') # 'auto' is usually good for scan plots unless a 1:1 scale is required

plt.tight_layout()

# --- Save to PNG using in-memory buffer for efficiency ---
buffer = io.BytesIO()
plt.savefig(buffer, format='PNG', dpi=2000) # Higher DPI for better image quality
buffer.seek(0)

# Load into PIL Image and save to file
img = Image.open(buffer)
img.save(output_scan_png_path)
print(f"Saved vertical scan graph to: {output_scan_png_path}")

plt.close(fig) # Close the Matplotlib figure
print("Done.")