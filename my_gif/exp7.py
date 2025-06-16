import matplotlib.pyplot as plt
import numpy as np


def get_pixel_position_of_origin(point = (0, 0)):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Set some example data and limits (adjust to match your actual plot)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 30)
    ax.grid(True)

    # Plot something so we have content
    x = np.linspace(0, 50, 100)
    y = np.sin(x / 5) * 10 + 10
    ax.plot(x, y, 'b-')

    # Get the transformation from data coordinates to display (pixel) coordinates
    # First draw the figure to make sure all elements are positioned
    fig.canvas.draw()

    # Get the transform from data to display coordinates
    transform = ax.transData

    # Transform the data point (0,0) to display coordinates
    display_coords = transform.transform(point)

    # Get the DPI and figure dimensions
    dpi = fig.get_dpi()
    width, height = fig.get_size_inches()

    # Calculate the pixel position relative to the figure
    pixel_x = display_coords[0] * dpi / width
    pixel_y = display_coords[1] * dpi / height

    print(f"Pixel position of (0,0): x={pixel_x:.1f}, y={pixel_y:.1f}")

    # Mark the (0,0) point for visual verification
    ax.plot(0, 0, 'ro', markersize=8, label='(0,0) point')
    ax.legend()

    plt.show()
    return pixel_x, pixel_y


# Example usage

point = (10, 55)
origin_x, origin_y = get_pixel_position_of_origin(point=point)
print(f"Final pixel coordinates: ({origin_x:.1f}, {origin_y:.1f})")























