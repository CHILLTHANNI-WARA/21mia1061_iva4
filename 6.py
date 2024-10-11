import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to detect scene cuts and create a summary
def detect_scene_cuts(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Initialize variables
    scene_cuts = []
    last_frame = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # If this is the first frame, just store it
        if last_frame is None:
            last_frame = gray_frame
            frame_count += 1
            continue

        # Calculate the absolute difference between the current frame and the last frame
        diff = cv2.absdiff(last_frame, gray_frame)

        # Threshold the difference to highlight changes
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Count the number of non-zero pixels (indicating a significant change)
        non_zero_count = np.count_nonzero(thresh)

        # If the number of changed pixels exceeds a certain threshold, it's a scene cut
        if non_zero_count > 5000:  # Adjust this threshold as needed
            scene_cuts.append(frame_count)
            cv2.putText(frame, "Scene Cut Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with Matplotlib
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title(f'Frame {frame_count}')
        plt.axis('off')
        plt.pause(0.01)  # Pause to update the plot

        # Update the last frame and frame count
        last_frame = gray_frame
        frame_count += 1

        # Break the loop on 'q' key press (this won't work in the plot window)
        if plt.waitforbuttonpress(0.01):  # Exit on any key press
            break

    # Release the video capture object
    cap.release()
    plt.close()

    # Create a summary of detected scene cuts
    summary = "Detected Scene Cuts at Frame Numbers: " + ", ".join(map(str, scene_cuts))
    print(summary)

    # Save the output summary to a text file
    output_summary_path = 'scene_cut_summary.txt'
    with open(output_summary_path, 'w') as f:
        f.write(summary)

# Replace 'path_to_your_video.mp4' with the path to your video file
video_path = '/Users/bharathvikram/Downloads/KFC Rockin Commercial 10 Sec (1).mp4'
detect_scene_cuts(video_path)
