import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate and plot color histograms
def plot_color_histogram(image_path, title):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image '{image_path}'.")
        return

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Calculate histograms for each color channel
    color_channels = ('r', 'g', 'b')
    hist_values = {}

    for i, color in enumerate(color_channels):
        hist_values[color] = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])

    # Plot the histograms
    plt.figure(figsize=(10, 5))
    for color in color_channels:
        plt.plot(hist_values[color], color=color, label=f'{color.upper()} Channel')
    plt.title(f'Color Histogram - {title}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.xlim([0, 256])
    plt.grid()
    plt.show()

# Upload your images
hard_cut_image_path = '/Users/bharathvikram/cut_frames/cut_frame_0024.png'  
soft_cut_image_path = '/Users/bharathvikram/cut_frames/cut_frame_0063.png' 

plot_color_histogram(hard_cut_image_path, "Hard Cut")
plot_color_histogram(soft_cut_image_path, "Soft Cut")
