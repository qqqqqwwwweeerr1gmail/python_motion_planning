import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# Your GifCreator class remains the same as in the previous, corrected response.
# It handles combining already correctly sized frames into a GIF.
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
        Combine frames into a GIF. This function assumes frames are already
        correctly sized and include axes/labels/title, with no excess padding.
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

def generate_and_save_frame_with_labels(frame_idx, output_filename,
                                        data_xmin=0, data_ymin=0, data_xmax=50, data_ymax=30):
    """
    Generates a Matplotlib plot frame with precise data limits,
    including axes, labels, and title, and saves it tightly cropped.
    """
    fig, ax = plt.subplots(figsize=(6, 4)) # Adjust figsize as needed for overall frame dimensions

    # Generate some dummy data that fits within the target range
    x_line = np.linspace(data_xmin, data_xmax, 100)
    y_line = np.sin(x_line / 5 + frame_idx * 0.5) * (data_ymax / 2.5) + (data_ymax / 2.5) # Ensure data is visible

    ax.plot(x_line, y_line, color='blue', linewidth=2)
    ax.scatter([data_xmin, data_xmax], [data_ymin, data_ymax], color='red', marker='o', s=100, zorder=5, label='Corners') # Mark corners

    ax.set_title(f"Animation Frame {frame_idx}")
    ax.set_xlabel("X-axis (Data Range)")
    ax.set_ylabel("Y-axis (Data Range)")

    # --- CRUCIAL: Set the axis limits EXACTLY to your data range ---
    ax.set_xlim(data_xmin, data_xmax)
    ax.set_ylim(data_ymin, data_ymax)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

    # --- This is the primary difference for including labels/title ---
    # `bbox_inches='tight'` will find the tightest box *around all figure elements*,
    # including the axes, ticks, labels, and title.
    # The `pad_inches=0.1` argument gives a tiny bit of breathing room if desired,
    # or set to `0` for absolute tightness.
    plt.savefig(output_filename, bbox_inches='tight', dpi=150, pad_inches=0.05) # Increased DPI for better quality
    plt.close(fig) # Always close figures to free memory


### Example Usage

if __name__ == "__main__":
    temp_frame_dir = "temp_frames_with_labels"
    if not os.path.exists(temp_frame_dir):
        os.makedirs(temp_frame_dir)

    frame_paths = []
    num_test_frames = 10
    for i in range(num_test_frames):
        frame_filename = os.path.join(temp_frame_dir, f"frame_{i:02d}.png")
        # Generate frames with axes perfectly set from 0,0 to 50,30, including labels
        generate_and_save_frame_with_labels(i, frame_filename,
                                            data_xmin=0, data_ymin=0,
                                            data_xmax=50, data_ymax=30)
        frame_paths.append(frame_filename)

    creator = GifCreator()
    for path in frame_paths:
        creator.add_frame(path)

    # _save_gif2 just combines the already correctly generated frames
    creator._save_gif2("animation_with_labels_and_axes.gif", loop_count=1) # Plays twice

    print(f"\nGenerated GIF: animation_with_labels_and_axes.gif")
    print(f"Frames saved in: {temp_frame_dir}")

    # Clean up
    import shutil
    shutil.rmtree(temp_frame_dir)
    print(f"\nCleaned up temporary frames in {temp_frame_dir}")























