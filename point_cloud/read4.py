import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData
from matplotlib.colors import ListedColormap


def create_walkability_map(ply_path, dpi=100, height_threshold=11.5):

    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    x = vertices['x']
    y = vertices['y']  # Height/attitude values
    z = vertices['z']

    xx = vertices.data['x']

    # Calculate grid dimensions
    min_x, max_x = np.min(x), np.max(x)
    min_z, max_z = np.min(z), np.max(z)
    min_y, max_y = np.min(y), np.max(y)

    grid_size = int((max_x - min_x)/dpi)

    # Create grid bins
    x_bins = np.arange(min_x, max_x + grid_size, grid_size)
    z_bins = np.arange(min_z, max_z + grid_size, grid_size)

    # Initialize height extremes
    grid_max = np.full((len(x_bins) - 1, len(z_bins) - 1), -np.inf)
    grid_min = np.full((len(x_bins) - 1, len(z_bins) - 1), np.inf)

    point_count = np.zeros((len(x_bins) - 1, len(z_bins) - 1), dtype=int)
    height_sum = np.zeros((len(x_bins) - 1, len(z_bins) - 1))
    height_avg = np.zeros((len(x_bins) - 1, len(z_bins) - 1), dtype=int)

    # Assign points to grid cells
    for xi, yi, zi in zip(x, y, z):
        x_idx = np.digitize(xi, x_bins) - 1
        z_idx = np.digitize(zi, z_bins) - 1

        # Handle edge cases
        x_idx = max(0, min(x_idx, grid_max.shape[0] - 1))
        z_idx = max(0, min(z_idx, grid_max.shape[1] - 1))

        # Update height extremes
        grid_max[x_idx, z_idx] = max(grid_max[x_idx, z_idx], yi)
        grid_min[x_idx, z_idx] = min(grid_min[x_idx, z_idx], yi)

        point_count[x_idx, z_idx] += 1
        height_sum[x_idx, z_idx] += yi

    # Calculate height variation per cell
    height_var = grid_max - grid_min
    height_avg = np.divide(height_sum, point_count,
                         out=np.full_like(height_sum, np.inf),  # inf for empty cells
                         where=point_count!=0)

    # Create walkability map
    walkable = np.ones_like(height_var, dtype=int)
    walkable[np.isinf(height_var)] = 0  # No data cells
    walkable[height_var > height_threshold] = 0  # High variation cells

    return walkable, (min_x, min_z), grid_size


def visualize_map(walkable_grid,dpi,max_height_var):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows Chinese font
    plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

    """Visualize the walkability grid with custom styling"""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Custom colormap: green for walkable, red for obstacles
    cmap = ListedColormap(['red', 'limegreen'])

    # Display the grid
    im = ax.imshow(walkable_grid.T, cmap=cmap, origin='lower')

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, walkable_grid.shape[0], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, walkable_grid.shape[1], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

    # Customize appearance
    ax.set_title(rf'{int(1000/dpi)}米一格,高度差{max_height_var}米', fontsize=16)
    ax.set_xlabel('X Grid Index', fontsize=12)
    ax.set_ylabel('Z Grid Index', fontsize=12)

    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='limegreen', label='Walkable Area'),
        Patch(facecolor='red', label='Obstacle/High Variation')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    plt.savefig(f'E:\GIT1\python_motion_planning\outputs\d3dpoint_{dpi}_{str(max_height_var).replace(""".""","__")}.png', dpi=150)  # Save the full frame PNG

    plt.show()


def save_walkability_data(walkable_grid, origin, grid_size, output_file):
    """Save walkability data to a text file"""
    with open(output_file, 'w') as f:
        f.write(f"Grid Origin (x,z): {origin[0]:.3f}, {origin[1]:.3f}\n")
        f.write(f"Grid Size: {grid_size:.3f}\n")
        f.write("Walkability Grid (1=walkable, 0=obstacle):\n")
        np.savetxt(f, walkable_grid, fmt='%d')


# Example usage
if __name__ == "__main__":
    # Parameters
    ply_file = "your_platform.ply"  # Replace with your PLY file
    cell_size = 1  # 10cm grid cells
    dpi = 200
    max_height_var = 0.5  # 5cm max variation

    # Process the PLY file
    walkable, origin, used_grid_size = create_walkability_map(
        ply_file, dpi=dpi, height_threshold=max_height_var
    )

    # Print summary
    print(f"Generated {walkable.shape[0]}x{walkable.shape[1]} grid")
    print(f"Walkable cells: {np.sum(walkable)}/{walkable.size} "
          f"({100 * np.sum(walkable) / walkable.size:.1f}%)")

    # Visualize
    visualize_map(walkable,dpi,max_height_var)

    # Save data
    save_walkability_data(walkable, origin, used_grid_size, "walkability_map.txt")