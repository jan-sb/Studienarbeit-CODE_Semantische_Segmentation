import cv2 as cv
import os

def convert_video_to_frames(video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Initialize a frame counter
    frame_count = 0

    # Read frames from the video and save as images
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Save the frame as an image in the output folder
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()

    print(f"Frames saved in {output_folder}")

# Specify the path to your .avi video file
video_path = 'testdata/instvid.avi'

# Specify the folder where you want to save the frames
output_folder = 'testdata/inst_picset'

# Call the function to convert the .avi video to frames
convert_video_to_frames(video_path, output_folder)
