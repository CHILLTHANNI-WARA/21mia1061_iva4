import cv2
import numpy as np
import os

# Function to detect hard and soft cuts in a video
def detect_cuts(video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    prev_frame = None  # To store the previous frame
    frame_count = 0  # Initialize frame counter

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Check if there is a previous frame to compare with
        if prev_frame is not None:
            # Calculate the absolute difference between the current and previous frames
            frame_diff = cv2.absdiff(prev_frame, gray)

            # Calculate the mean value of the difference
            mean_diff = np.mean(frame_diff)

            # Thresholds for cut detection
            hard_cut_threshold = 30  # You can adjust this value
            soft_cut_threshold = 10   # You can adjust this value

            # Determine if the difference indicates a hard or soft cut
            if mean_diff > hard_cut_threshold:
                cut_type = "Hard Cut"
                print(f"Hard Cut detected at frame {frame_count}")
            elif mean_diff > soft_cut_threshold:
                cut_type = "Soft Cut"
                print(f"Soft Cut detected at frame {frame_count}")
            else:
                cut_type = "No Cut"

            # Save frames where cuts are detected
            output_filename = os.path.join(output_folder, f'cut_frame_{frame_count:04d}.png')
            cv2.imwrite(output_filename, frame)
        
        # Store the current frame as the previous frame for the next iteration
        prev_frame = gray

    # Release the video capture object
    cap.release()
    print(f"Finished processing {frame_count} frames.")

# Replace 'path_to_your_video.mp4' with the path to your video file
video_path = '/Users/bharathvikram/Downloads/KFC Rockin Commercial 10 Sec (1).mp4'
output_folder = 'cut_frames'  # Specify the output folder for cut frames
detect_cuts(video_path, output_folder)
