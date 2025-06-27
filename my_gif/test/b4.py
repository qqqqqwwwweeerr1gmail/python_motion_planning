
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO

# --- YOUR EXISTING CLASSES (AS PROVIDED IN THE PROBLEM DESCRIPTION) ---
from math import sqrt
from scipy.spatial import cKDTree

class Node:
    def __init__(self, pos, parent, cost, heuristic):
        self.x = pos[0]
        self.y = pos[1]
        self.pos = pos
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic

class Env:
    def __init__(self, x_range_or_bounds, y_range_or_bounds):
        self.x_range = x_range_or_bounds
        self.y_range = y_range_or_bounds

class Grid(Env):
    def __init__(self, x_range: int, y_range: int) -> None:
        super().__init__(x_range, y_range)
        self.motions = [Node((-1, 0), None, 1, None), Node((-1, 1),  None, sqrt(2), None),
                        Node((0, 1),  None, 1, None), Node((1, 1),   None, sqrt(2), None),
                        Node((1, 0),  None, 1, None), Node((1, -1),  None, sqrt(2), None),
                        Node((0, -1), None, 1, None), Node((-1, -1), None, sqrt(2), None)]
        self.obstacles = None
        self.obstacles_tree = None
        self.init()

    def init(self) -> None:
        x, y = self.x_range, self.y_range
        obstacles = set()
        for i in range(x):
            obstacles.add((i, 0))
            obstacles.add((i, y - 1))
        for i in range(y):
            obstacles.add((0, i))
            obstacles.add((x - 1, i))
        self.update(obstacles)

    def update(self, obstacles):
        self.obstacles = obstacles
        self.obstacles_tree = cKDTree(np.array(list(obstacles)))
# --- END OF YOUR EXISTING CLASSES ---


class Plot:
    def __init__(self, start, goal, env: Env):
        self.start = Node(start, start, 0, 0)
        self.goal = Node(goal, goal, 0, 0)
        self.env = env

        self.fig = plt.figure("planning")
        self.ax = self.fig.add_subplot()

        if hasattr(self.env, 'x_range') and hasattr(self.env, 'y_range'):
            x_plot_min, x_plot_max = 0, self.env.x_range
            y_plot_min, y_plot_max = 0, self.env.y_range
            print(f"Plot limits set from Grid's x_range={self.env.x_range}, y_range={self.env.y_range}")
        elif hasattr(self.env, 'x_bounds') and hasattr(self.env, 'y_bounds'):
            x_plot_min, x_plot_max = self.env.x_bounds[0], self.env.x_bounds[1]
            y_plot_min, y_plot_max = self.env.y_bounds[0], self.env.y_bounds[1]
            print(f"Plot limits set from Env's x_bounds={self.env.x_bounds}, y_bounds={self.env.y_bounds}")
        else:
            x_plot_min, x_plot_max = 0, 50
            y_plot_min, y_plot_max = 0, 30
            print("Warning: Environment attributes for plot limits not found. Using default (0,50)x(0,30).")

        self.ax.set_xlim(x_plot_min, x_plot_max)
        self.ax.set_ylim(y_plot_min, y_plot_max)
        self.ax.grid(True, linestyle=':', alpha=0.6)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title("Motion Planning Visualization")
        self.ax.set_xlabel("X-coordinate")
        self.ax.set_ylabel("Y-coordinate")

        self.frames = []

    def get_pixel_coords(self, data_point):
        self.fig.canvas.draw()
        display_coords = self.ax.transData.transform(data_point)
        fig_width_pixels, fig_height_pixels = self.fig.canvas.get_width_height()
        pixel_x = int(display_coords[0])
        pixel_y = int(fig_height_pixels - display_coords[1])
        return (pixel_x, pixel_y)

    def draw_frame(self, frame_idx: int):
        for line in self.ax.lines:
            line.remove()
        for patch in self.ax.patches:
            patch.remove()
        for text in self.ax.texts:
            text.remove()
        if self.ax.collections:
            for collection in self.ax.collections:
                collection.remove()

        x_data = np.linspace(self.ax.get_xlim()[0], self.ax.get_xlim()[1], 100)
        y_data = np.sin(x_data / 5 + frame_idx * 0.5) * (self.ax.get_ylim()[1] - self.ax.get_ylim()[0]) / 3 + \
                 (self.ax.get_ylim()[0] + self.ax.get_ylim()[1]) / 2
        self.ax.plot(x_data, y_data, color='blue', linewidth=2, label='Dynamic Path')

        self.ax.plot(self.start.x, self.start.y, 'go', markersize=10, label='Start')
        self.ax.plot(self.goal.x, self.goal.y, 'ro', markersize=10, label='Goal')

        self.ax.text(self.ax.get_xlim()[0] + 1, self.ax.get_ylim()[1] - 1,
                     f"Frame: {frame_idx}", color='black', fontsize=10,
                     ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        if self.ax.get_legend():
            self.ax.get_legend().remove()
        self.ax.legend(loc='lower right')

        self.fig.tight_layout()

        buf = BytesIO()
        self.fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)

        img = Image.open(buf)
        img.load()
        self.frames.append(img)

        buf.close()

    def _save_gif1(self, gif_path: str, loop_count: int = None):
        """
        Save frames as a GIF without any cropping.

        Args:
            gif_path: Output file path for the GIF.
            loop_count: GIF loop count (None=plays once, 0=loops infinitely, N>0=loops N+1 times).
        """
        if not self.frames:
            print("No frames to save for GIF. Call draw_frame() multiple times first.")
            return

        # Ensure all frames are in RGB mode for GIF compatibility
        processed_frames = []
        for frame in self.frames:
            if frame.mode == 'RGBA':
                # Create a white background for alpha blending if original has transparency
                background = Image.new('RGB', frame.size, (255, 255, 255))
                background.paste(frame, mask=frame.split()[3])
                processed_frames.append(background)
            else:
                processed_frames.append(frame.convert('RGB'))

        if not processed_frames:
            print("Error: No processed frames to save for GIF.")
            return

        save_kwargs = {
            "save_all": True,
            "append_images": processed_frames[1:] if len(processed_frames) > 1 else [],
            "duration": 100,  # Milliseconds per frame
            "optimize": True  # Optimize GIF palette
        }

        if loop_count is not None:
            save_kwargs["loop"] = loop_count

        processed_frames[0].save(gif_path, **save_kwargs)

        if loop_count is None:
            print(f"GIF saved to {gif_path} (plays once and stops).")
        elif loop_count == 0:
            print(f"GIF saved to {gif_path} (loops infinitely).")
        else:
            print(f"GIF saved to {gif_path} (plays {loop_count + 1} times).")


    def _save_gif2(self, gif_path: str, loop_count: int = None,
                   point1=(0, 0), point2=(50, 30)):
        if not self.frames:
            print("No frames to save. Call draw_frame() multiple times first.")
            return

        self.fig.canvas.draw()
        fig_width_pixels, fig_height_pixels = self.fig.canvas.get_width_height()

        crop_data_xmin = min(point1[0], point2[0])
        crop_data_ymin = min(point1[1], point2[1])
        crop_data_xmax = max(point1[0], point2[0])
        crop_data_ymax = max(point1[1], point2[1])

        pixel_bl_data = self.ax.transData.transform((crop_data_xmin, crop_data_ymin))
        pixel_tr_data = self.ax.transData.transform((crop_data_xmax, crop_data_ymax))

        left = int(min(pixel_bl_data[0], pixel_tr_data[0]))
        right = int(max(pixel_bl_data[0], pixel_tr_data[0]))

        upper = int(fig_height_pixels - max(pixel_bl_data[1], pixel_tr_data[1]))
        lower = int(fig_height_pixels - min(pixel_bl_data[1], pixel_tr_data[1]))

        frame_width, frame_height = self.frames[0].size
        left = max(0, min(left, frame_width))
        right = max(0, min(right, frame_width))
        upper = max(0, min(upper, frame_height))
        lower = max(0, min(lower, frame_height))

        if left >= right or upper >= lower:
            print \
                (f"Warning: Calculated crop region resulted in zero or negative size: ({left}, {upper}, {right}, {lower}). Cannot crop.")
            return

        processed_frames = []
        for frame in self.frames:
            if frame.mode == 'RGBA':
                background = Image.new('RGB', frame.size, (255, 255, 255))
                background.paste(frame, mask=frame.split()[3])
                img = background
            else:
                img = frame.convert('RGB')

            cropped_img = img.crop((left, upper, right, lower))
            processed_frames.append(cropped_img)

        if not processed_frames:
            print("Error: No processed frames to save after cropping.")
            return

        save_kwargs = {
            "save_all": True,
            "append_images": processed_frames[1:] if len(processed_frames) > 1 else [],
            "duration": 100,
            "optimize": True
        }
        if loop_count is not None:
            save_kwargs["loop"] = loop_count

        processed_frames[0].save(gif_path, **save_kwargs)
        print(f"GIF saved to {gif_path}")


# --- Example Usage ---
if __name__ == '__main__':
    start = (5 ,5)
    goal = (45 ,25)
    grid_env = Grid(51, 31)
    env = grid_env

    plot = Plot(start, goal, env)

    num_animation_frames = 20
    print(f"Generating {num_animation_frames} frames...")
    for i in range(num_animation_frames):
        plot.draw_frame(i)
    print(f"\nCaptured {len(plot.frames)} frames.")

    # --- New: Calling _save_gif1 for uncropped GIF ---
    plot._save_gif1("output_animation_no_crop.gif", loop_count=0)
    print("\n--- Uncropped GIF created ---")

    # --- Existing: Calling _save_gif2 for cropped GIF ---
    plot._save_gif2("output_animation_cropped.gif",
                    loop_count=0,
                    point1=(0, 0),
                    point2=(50, 30))
    print("\n--- Cropped GIF created ---")

    plt.close(plot.fig)
    print("\nScript finished.")













