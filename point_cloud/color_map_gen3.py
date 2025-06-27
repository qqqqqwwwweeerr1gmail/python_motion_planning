import numpy as np
from plyfile import PlyData
from PIL import Image, ImageDraw

# Load PLY file


ply_file = r"E:\GIT1\python_motion_planning\point_cloud\Depth2PC\workspace\0_pc_world_space.ply"  # Replace with your PLY file
ply_file = "your_platform.ply"  # Replace with your PLY file
ply_file = r"E:\GIT1\python_motion_planning\point_cloud\Depth2PC\workspace\0_pc_camera_space.ply"  # Replace with your PLY file
ply_file = r"/point_cloud/datas/total.ply"  # Replace with your PLY file


ply = PlyData.read(ply_file)
v = ply['vertex']
x, y, z = v['x'], v['y'], v['z']
rgb = np.vstack((v['red'], v['green'], v['blue'])).T  # Shape: (N, 3)

# Parameters
output_width = 2000  # Width of output image
output_height = 2000  # Height (adjust based on Y-range)
point_radius = 3  # Size of each point (adjust as needed)

# Normalize coordinates
y_min, y_max = y.min(), y.max()
x_min, x_max = x.min(), x.max()
z_min, z_max = z.min(), z.max()

# Create blank image
img = Image.new('RGB', (output_width, output_height), (0, 0, 0))
draw = ImageDraw.Draw(img)

# Bin points into vertical columns
x_bins = np.linspace(x_min, x_max, output_width)
bin_indices = np.digitize(x, x_bins) - 1

# For each vertical column
for col in range(output_width):
    # Get points in this column
    mask = (bin_indices == col)
    if not np.any(mask):
        continue

    # Sort points by height (Z)
    sorted_idx = np.argsort(z[mask])
    z_col = z[mask][sorted_idx]
    rgb_col = rgb[mask][sorted_idx]

    # Map Z to image rows
    rows = ((z_col - z_min) / (z_max - z_min) * (output_height - 1)).astype(int)

    # Draw larger points
    for i, row in enumerate(rows):
        if 0 <= row < output_height:
            # Draw a circle for each point
            x0, y0 = col - point_radius, row - point_radius
            x1, y1 = col + point_radius, row + point_radius
            draw.ellipse([x0, y0, x1, y1], fill=tuple(rgb_col[i]))

# Save as PNG
img.save("vertical_scan_large_points.png")