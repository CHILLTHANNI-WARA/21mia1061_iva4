import cv2
import numpy as np

# Function for foreground detection and object tracking
def foreground_detection_and_tracking(video_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Create the background subtractor object
    backSub = cv2.createBackgroundSubtractorMOG2()

    # Get the width and height of the frames to create the output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Apply the background subtractor to get the foreground mask
        fg_mask = backSub.apply(frame)

        # Perform morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)

        # Find contours in the foreground mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around detected objects
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Adjust the area threshold as needed
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box

        # Write the processed frame to the output video
        out.write(frame)

        # Display the original frame and the foreground mask (optional)
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Foreground Mask', fg_mask)

        # Break the loop on 'q' key press
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Replace 'path_to_your_video.mp4' with the path to your input video file
# Specify the output video path
video_path = '/Users/bharathvikram/Downloads/KFC Rockin Commercial 10 Sec (1).mp4'
output_video_path = 'output_foreground_tracking.avi'
foreground_detection_and_tracking(video_path, output_video_path)

print(f"Processed video saved as: {output_video_path}")
