import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# Your existing GifCreator class (no changes needed here for cropping logic)
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

    def _save_gif2(self, gif_path: str, loop_count: int = None):
        '''
        Combine frames into a GIF.
        This function now assumes frames are already correctly cropped by the source.

        Args:
            gif_path (str): The path to save the GIF.
            loop_count (int, optional):
                - None (default): GIF plays exactly once and stops.
                - 0: GIF loops infinitely.
                - N > 0: GIF plays once, then loops N more times (total N+1 plays).
        '''
        if not self.frames:
            print("No frames to save.")
            return

        processed_frames = []
        for frame in self.frames:
            if frame.mode == 'RGBA':
                background = Image.new('RGB', frame.size, (255, 255, 255))
                background.paste(frame, mask=frame.split()[3])
                processed_frames.append(background)
            else:
                processed_frames.append(frame.convert('RGB'))

        if not processed_frames:
            print("Error: No processed frames to save.")
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

# --- Function to generate and save a single frame with PRECISE axis limits ---
def generate_and_save_frame_precise(frame_idx, output_filename,
                                    data_xmin=0, data_ymin=0, data_xmax=50, data_ymax=30):
    """
    Generates a Matplotlib plot frame with axes set precisely to the desired data range.
    """
    fig, ax = plt.subplots(figsize=(6, 4)) # Adjust figsize for desired output resolution

    # Generate some dummy data that fits within the target range
    x = np.random.uniform(data_xmin, data_xmax, 20)
    y = np.random.uniform(data_ymin, data_ymax, 20)
    # Or for a continuous line:
    x_line = np.linspace(data_xmin, data_xmax, 100)
    y_line = np.sin(x_line / 5 + frame_idx * 0.5) * (data_ymax / 2 - data_ymin / 2) + (data_ymax + data_ymin) / 2

    ax.plot(x_line, y_line, color='blue', linewidth=2)
    ax.scatter(x, y, color='red', s=50, alpha=0.7)


    ax.set_title(f"Frame {frame_idx} (Cropped at Source)")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    # --- CRUCIAL STEP: Set the axis limits EXACTLY to your data range ---
    ax.set_xlim(data_xmin, data_xmax)
    ax.set_ylim(data_ymin, data_ymax)
    ax.grid(True, linestyle=':', alpha=0.6)

    # Use tight_layout and bbox_inches='tight' to remove extra whitespace *around* the plot,
    # ensuring the saved image is just the plot area itself.
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', dpi=100) # Adjust DPI for quality
    plt.close(fig) # Always close figures to free memory

# --- Example Usage ---
if __name__ == "__main__":
    temp_frame_dir = "temp_precise_frames"
    if not os.path.exists(temp_frame_dir):
        os.makedirs(temp_frame_dir)

    frame_paths = []
    num_test_frames = 10
    for i in range(num_test_frames):
        frame_filename = os.path.join(temp_frame_dir, f"frame_{i:02d}.png")
        # Generate frames with axes perfectly set from 0,0 to 50,30
        generate_and_save_frame_precise(i, frame_filename,
                                        data_xmin=0, data_ymin=0,
                                        data_xmax=50, data_ymax=30)
        frame_paths.append(frame_filename)

    creator = GifCreator()
    for path in frame_paths:
        creator.add_frame(path)

    # Now, save the GIF. No cropping parameters needed in _save_gif2
    # because the frames are already generated with the correct bounds.
    creator._save_gif2("precise_cropped_animation.gif", loop_count=1) # Plays twice

    print(f"\nGenerated GIF: precise_cropped_animation.gif")
    print(f"Frames saved in: {temp_frame_dir}")

    # Clean up
    import shutil
    shutil.rmtree(temp_frame_dir)
    print(f"\nCleaned up temporary frames in {temp_frame_dir}")























