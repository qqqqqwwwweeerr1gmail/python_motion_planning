









import os
from PIL import Image, ImageDraw

class GifCreator:
    def __init__(self):
        self.frames = []

    def add_frame(self, image_path: str):
        try:
            img = Image.open(image_path)
            self.frames.append(img)
        except FileNotFoundError:
            print(f"Error: Frame image not found at {image_path}")
        except Exception as e:
            print(f"Error adding frame {image_path}: {e}")

    def _save_gif2(self, gif_path: str, loop_count: int = None,
                   crop_point1= (0,0), crop_point2=(50,30)):

        if not self.frames:
            print("No frames to save.")
            return

        processed_frames = []
        for frame in self.frames:
            # Convert to RGB (handling transparency if needed)
            if frame.mode == 'RGBA':
                # Create a white background for alpha blending before cropping
                background = Image.new('RGB', frame.size, (255, 255, 255))
                background.paste(frame, mask=frame.split()[3])
                img = background
            else:
                img = frame.convert('RGB')

            width, height = img.size

            pixel_left =
            pixel_right =
            pixel_upper =
            pixel_lower =


            if pixel_upper > pixel_lower:
                pixel_upper, pixel_lower = pixel_lower, pixel_upper # Swap if inverted by calculation

            cropped_img = img.crop((pixel_left, pixel_upper, pixel_right, pixel_lower))
            processed_frames.append(cropped_img)

        # Ensure there's at least one frame to save after processing
        if not processed_frames:
            print("Error: No processed frames to save after cropping.")
            return

        save_kwargs = {
            "save_all": True,
            "append_images": processed_frames[1:] if len(processed_frames) > 1 else [],
            "duration": 100,  # Milliseconds per frame
            "optimize": True
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

# --- Example of how you would generate frames *before* passing to GifCreator ---
# This part assumes you're generating your frames using Matplotlib
# and they currently have the larger 0,-5 to 55,35 range.

import matplotlib.pyplot as plt
import numpy as np

def generate_sample_frame(frame_idx: int, output_filename: str):
    """
    Generates a sample Matplotlib plot that explicitly spans the full data range
    from (0,-5) to (55,35), representing the content that _save_gif2 will receive.
    It also visually marks the desired crop area (0,0) to (50,30).
    """
    fig, ax = plt.subplots(figsize=(8, 6)) # Adjust figsize for desired input frame resolution

    # Generate some data that spans the full original data range and moves
    x_data = np.linspace(0, 55, 200)
    y_data = np.sin(x_data / 7 + frame_idx * 0.2) * 15 + 15 # Data ranges roughly -5 to 35

    ax.plot(x_data, y_data, color='blue', linewidth=2, label='Sample Data')

    # Plot a red rectangle to visually represent the desired crop area (0,0) to (50,30)
    # This helps verify the cropping
    ax.add_patch(plt.Rectangle((0, 0), 50, 30,
                               edgecolor='red', facecolor='none',
                               linestyle='--', linewidth=2, label='Desired Crop Area'))


    ax.set_title(f"Source Frame {frame_idx} (Full Extent)")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    # --- Set the axes limits for this *source* image to match the "0,-5 to 55,35" range ---
    ax.set_xlim(0, 55)
    ax.set_ylim(-5, 35)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

    # Ensure layout is tight for the saving process to minimize unnecessary borders
    plt.tight_layout()

    # Save the figure. Increase dpi for higher quality output frames.
    plt.savefig(output_filename, dpi=100) # Save at a resolution that suits your needs
    plt.close(fig) # Close the figure to free memory

# --- Main execution block for testing ---
if __name__ == "__main__":
    temp_frame_dir = "temp_rectangular_cropped_gif_frames"
    if not os.path.exists(temp_frame_dir):
        os.makedirs(temp_frame_dir)

    frame_paths = []
    num_test_frames = 10 # Generate more frames for a smoother GIF
    for i in range(num_test_frames):
        frame_filename = os.path.join(temp_frame_dir, f"frame_{i:02d}.png")
        # Generate frames that span the full original data range for _save_gif2 to crop
        generate_sample_frame(i, frame_filename)
        frame_paths.append(frame_filename)

    gif_creator = GifCreator()
    for path in frame_paths:
        gif_creator.add_frame(path)

    # Now, save the GIF using _save_gif2 with the specified rectangular cropping parameters
    # The original images cover data from (0,-5) to (55,35).
    # We want to crop to the data region from (0,0) to (50,30).
    gif_creator._save_gif2(
        "rectangular_cropped_animation.gif",
        loop_count=1, # Play twice (initial play + 1 loop)
        crop_point1=(0, 0), crop_point2=(50, 30),
    )

    # Clean up the temporary frame directory
    import shutil
    shutil.rmtree(temp_frame_dir)
    print(f"\nCleaned up temporary frames in {temp_frame_dir}")













