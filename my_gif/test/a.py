from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from srca.python_motion_planning.utils.environment.env import Env, Grid, Map, Node
import os

class Plot:
    def __init__(self, start, goal, env: Env):
        self.start = Node(start, start, 0, 0)
        self.goal = Node(goal, goal, 0, 0)
        self.env = env
        self.fig = plt.figure("planning")
        self.ax = self.fig.add_subplot()
        self.frames = []  # Store frames for GIF

    def get_pixel_coords(self, data_point):
        """Convert data coordinates to pixel coordinates in the saved image"""
        # First ensure the figure is drawn
        self.fig.canvas.draw()

        # Get the transform from data to display coordinates
        display_coords = self.ax.transData.transform(data_point)

        # Convert display coordinates to image pixel coordinates
        # Note: This assumes the saved image matches the display size
        return (int(display_coords[0]), int(display_coords[1]))

    def _save_gif2(self, gif_path: str, loop_count: int = None,
                   point1=(0, 0), point2=(50, 30)):
        """
        Save frames as GIF, cropping to rectangle defined by two data points.

        Args:
            gif_path: Output file path
            loop_count: GIF loop count (None=once, 0=infinite)
            point1: First corner point in data coordinates (x,y)
            point2: Opposite corner point in data coordinates (x,y)
        """
        if not self.frames:
            print("No frames to save.")
            return

        # Get pixel coordinates for all four corners
        x_coords = sorted([point1[0], point2[0]])
        y_coords = sorted([point1[1], point2[1]])

        # Calculate the four corners of the rectangle
        corners = [
            (x_coords[0], y_coords[0]),  # bottom-left
            (x_coords[1], y_coords[0]),  # bottom-right
            (x_coords[1], y_coords[1]),  # top-right
            (x_coords[0], y_coords[1])  # top-left
        ]

        # Convert all corners to pixel coordinates
        pixel_corners = [self.get_pixel_coords(c) for c in corners]

        # Find the bounding box
        left = min(p[0] for p in pixel_corners)
        right = max(p[0] for p in pixel_corners)
        upper = min(p[1] for p in pixel_corners)
        lower = max(p[1] for p in pixel_corners)

        processed_frames = []
        for frame in self.frames:
            # Handle transparency if needed
            if frame.mode == 'RGBA':
                background = Image.new('RGB', frame.size, (255, 255, 255))
                background.paste(frame, mask=frame.split()[3])
                img = background
            else:
                img = frame.convert('RGB')

            # Perform the crop
            cropped_img = img.crop((left, upper, right, lower))
            processed_frames.append(cropped_img)

        # Save GIF
        save_kwargs = {
            "save_all": True,
            "append_images": processed_frames[1:],
            "duration": 100,
            "optimize": True
        }
        if loop_count is not None:
            save_kwargs["loop"] = loop_count

        processed_frames[0].save(gif_path, **save_kwargs)
        print(f"GIF saved to {gif_path}")


if __name__ == '__main__':
    start = (5,5)
    goal = (45,25)
    grid_env = Grid(51, 31)
    env = grid_env
    plot = Plot(start, goal, env)
    plot._save_gif2("output.gif",
                    point1=(0, 0),  # Bottom-left of your area
                    point2=(50, 30))  # Top-right of your area




















