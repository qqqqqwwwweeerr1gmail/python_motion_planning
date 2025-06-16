import os
from PIL import Image

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

    def _save_gif(self, gif_path: str, loop_count: int = None):
        '''
        Combine frames into a GIF.

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

        save_kwargs = {
            "save_all": True,
            "append_images": processed_frames[1:] if len(processed_frames) > 1 else [],
            "duration": 100, # Milliseconds per frame
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

# --- Example Usage ---
if __name__ == "__main__":
    if not os.path.exists("temp_frames"):
        os.makedirs("temp_frames")

    for i, color in enumerate([(255,0,0), (0,255,0), (0,0,255)]):
        img = Image.new('RGB', (100, 100), color)
        img.save(f"temp_frames/frame_{i}.png")

    creator = GifCreator()
    creator.add_frame("temp_frames/frame_0.png")
    creator.add_frame("temp_frames/frame_1.png")
    creator.add_frame("temp_frames/frame_2.png")

    print("\n--- Testing GIF loops ---")

    # To play the GIF exactly once:
    creator._save_gif("output_gif_once.gif", loop_count=None)

    # To play the GIF infinitely:
    creator._save_gif("output_gif_infinite.gif", loop_count=0)

    # To play the GIF exactly twice (original play + 1 loop):
    creator._save_gif("output_gif_twice.gif", loop_count=1)

    # To play the GIF exactly three times (original play + 2 loops):
    creator._save_gif("output_gif_thrice.gif", loop_count=2)

    # Clean up dummy frames
    for i in range(3):
        os.remove(f"temp_frames/frame_{i}.png")
    os.rmdir("temp_frames")
    print("\nTemporary frames cleaned up.")