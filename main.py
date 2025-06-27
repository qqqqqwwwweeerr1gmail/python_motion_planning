import math
import sys, os
import time

import numpy as np
from plyfile import PlyData
from PIL import Image, ImageDraw

from tools.gen_block import gen_block

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python_motion_planning import *
import matplotlib
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP


def plan(grid_env,algorithm='AStar',start=(5, 5), goal=(45, 25)):
    matplotlib.use('Agg')

    file_abs = os.path.dirname(__file__)



    output_abs = str(os.path.join(file_abs, 'outputs\output.gif'))
    print(f"output_abs Root: {output_abs}")

    output_gif = Path(output_abs)

    if output_gif.exists():
        output_gif.unlink()  # Delete the file

    algorithm_mapping = {
        'AStar': AStar,
        'DStar': DStar,
        'DStarLite': DStarLite,
        'Dijkstra': Dijkstra,
        'GBFS': GBFS,
        'JPS': JPS,
        'ThetaStar': ThetaStar,
        'LazyThetaStar': LazyThetaStar,
        'SThetaStar': SThetaStar,
        'LPAStar': LPAStar,
        'VoronoiPlanner': VoronoiPlanner,
    }

    plt = algorithm_mapping[algorithm](start=start, goal=goal, env=grid_env)

    plt.run()

    def wait_for_file(path, timeout=10, check_interval=0.5):
        """Wait for a file to exist and be non-empty."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if path.exists() and path.stat().st_size > 0:
                return True
            time.sleep(check_interval)
        return False

    if wait_for_file(output_gif):
        with output_gif.open('rb') as f:
            bin_gif = f.read()
        return {"res_code": "200", "res_content": bin_gif}
    else:
        return {"res_code": "500", "res_content": "GIF generation failed!"}


import matplotlib.pyplot as plt

def gen_base_plotEnv(grid_env,start=(-1, -1),goal=(-1, -1)) -> None:
    pllt = Plot(start, goal,grid_env)
    img_binary = pllt.plotEnv('myframe')
    return img_binary

def rescale(grid_env,point):
    x_scaled = (grid_env.x_range-2) / 100 * point[0]
    y_scaled = (grid_env.y_range - 2) / 100 * point[1]
    x_scaled = int(Decimal(str(x_scaled)).quantize(Decimal('1.'), rounding=ROUND_HALF_UP))
    y_scaled = int(Decimal(str(y_scaled)).quantize(Decimal('1.'), rounding=ROUND_HALF_UP))
    return (x_scaled, y_scaled)



def get_scan_graph_from_3d_point():
    ply_file = r"/point_cloud/datas/total.ply"  # Replace with your PLY file

    file_abs = os.path.dirname(__file__)
    ply_file = str(os.path.join(file_abs, r'point_cloud\datas\total.ply'))


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
    # img.save("vertical_scan_large_points.png")
    img_bytes = img.tobytes()
    return img_bytes


if __name__ == '__main__':

    # # Create environment with custom obstacles
    grid_env = Grid(102, 102)
    # grid_env = Grid(202, 202)

    obstacles = grid_env.obstacles


    starts = [10,30,50,70]
    ends = [10,30,50,70]
    # ends = [20,40,60,80]

    for start in starts:
        for end in ends:
            blocks = gen_block(start=(start,end),end= (start+10,end+10))
            [obstacles.add(i) for i in blocks]
        #     break
        # break

    grid_env.update(obstacles)
    print(grid_env)
    #
    out_gif_binnary = plan(grid_env, algorithm='AStar', start=(5, 5), goal=(45, 25))
    print(out_gif_binnary)



    img_binary = gen_base_plotEnv(grid_env)
    img_binary = gen_base_plotEnv(grid_env,start=(5, 5))
    print(img_binary)

    point  = (2.3,4.5)
    point = rescale(grid_env,point)
    print(point)

    img_bytes = get_scan_graph_from_3d_point()
    print(img_bytes)






