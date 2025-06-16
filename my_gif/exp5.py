import matplotlib.pyplot as plt
import numpy as np

# --- 1. Function to generate your Matplotlib frames (source images for cropping) ---
def generate_matplotlib_frame(frame_idx: int, output_filename: str):
    """
    Generates a sample Matplotlib plot.
    This plot will intentionally display a wider data range than what you want to crop.
    """
    # Define the data range that *this specific Matplotlib plot will display*.
    # This range includes the padding Matplotlib adds or you explicitly set.
    # YOUR PREVIOUS STATEMENT: "the plt show from 0,-5 to 55,35"
    plot_data_xmin = 0
    plot_data_ymin = -5
    plot_data_xmax = 55
    plot_data_ymax = 35

    fig, ax = plt.subplots(figsize=(8, 6)) # Choose a suitable size for the raw PNG

    # Plot some data, some of which might fall within your desired (0,0)-(50,30) crop,
    # and some might be outside, but within the (0,-5)-(55,35) display range.
    x_data = np.linspace(plot_data_xmin, plot_data_xmax, 200)
    y_data = np.sin(x_data / 5 + frame_idx * 0.2) * 15 + 15 # Example data

    ax.plot(x_data, y_data, color='blue', linewidth=2, label='Animated Data')

    # Add visual markers for the desired crop area (0,0) to (50,30)
    # These will help you verify the crop is correct on the final GIF.
    ax.plot([0, 50, 50, 0, 0], [0, 0, 30, 30, 0], 'r--', linewidth=2, label='Desired Crop Region (Data)')
    ax.scatter([0, 50], [0, 30], color='green', marker='X', s=200, zorder=5, label='Crop Corners')

    # Add text/labels that might get cut off
    ax.set_title(f"Original Frame {frame_idx} (Full View)")
    ax.set_xlabel("X-Axis Label (e.g., beyond 50)")
    ax.set_ylabel("Y-Axis Label (e.g., below 0)")

    # --- IMPORTANT: Set the limits of this Matplotlib plot ---
    # These are the `original_img_data_...` values you'll pass to _save_gif2.
    ax.set_xlim(plot_data_xmin, plot_data_xmax)
    ax.set_ylim(plot_data_ymin, plot_data_ymax)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

    plt.tight_layout() # Adjust layout to ensure labels fit initially
    plt.savefig(output_filename, dpi=150) # Save the full frame PNG
    plt.close(fig) # Close the figure to free memory


# --- 2. Main execution to generate frames and create the GIF ---
if __name__ == "__main__":
    temp_frame_dir = "temp_raw_matplotlib_frames"
    if not os.path.exists(temp_frame_dir):
        os.makedirs(temp_frame_dir)

    frame_paths = []
    num_test_frames = 10
    for i in range(num_test_frames):
        frame_filename = os.path.join(temp_frame_dir, f"frame_{i:02d}.png")
        # Generate the raw Matplotlib frames, which have your "0,-5 to 55,35" data range
        generate_matplotlib_frame(i, frame_filename)
        frame_paths.append(frame_filename)

    gif_creator = GifCreator()
    for path in frame_paths:
        gif_creator.add_frame(path)

    # Now, save the GIF. Here's where you define the precise crop:
    # `crop_data_...`: This is what you want the final GIF corners to be (your data range).
    # `original_img_data_...`: This is the data range *currently displayed* in the PNGs you just generated.
    gif_creator._save_gif2(
        "final_cropped_gif_data_corners.gif",
        loop_count=1, # Play twice
        crop_data_xmin=0, crop_data_ymin=0,
        crop_data_xmax=50, crop_data_ymax=30,
        original_img_data_xmin=0,       # The x-min of your raw Matplotlib plot
        original_img_data_ymin=-5,      # The y-min of your raw Matplotlib plot
        original_img_data_xmax=55,      # The x-max of your raw Matplotlib plot
        original_img_data_ymax=35       # The y-max of your raw Matplotlib plot
    )

    print(f"\nGenerated GIF: final_cropped_gif_data_corners.gif")
    print(f"Temporary raw frames saved in: {temp_frame_dir}")

    # Clean up
    import shutil
    shutil.rmtree(temp_frame_dir)
    print(f"\nCleaned up temporary frames in {temp_frame_dir}")