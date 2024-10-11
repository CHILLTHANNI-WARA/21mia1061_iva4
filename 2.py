import cv2
import numpy as np
import os

# Function to apply Sobel edge detection and save output frames
def sobel_edge_detection(video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0  # Initialize frame counter

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Sobel edge detection
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)  # Sobel in X direction
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)  # Sobel in Y direction

        # Calculate the magnitude of gradients
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_magnitude = cv2.convertScaleAbs(sobel_magnitude)

        # Construct the output filename
        output_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.png')
        
        # Save the edge-detected frame
        cv2.imwrite(output_filename, sobel_magnitude)
        
        frame_count += 1  # Increment frame counter

    # Release the video capture object
    cap.release()
    print(f"Saved {frame_count} frames to '{output_folder}'.")

# Replace 'path_to_your_video.mp4' with the path to your video file
video_path = '/Users/bharathvikram/Downloads/KFC Rockin Commercial 10 Sec (1).mp4'
output_folder = 'output_frames'  # Specify the output folder name
sobel_edge_detection(video_path, output_folder)
