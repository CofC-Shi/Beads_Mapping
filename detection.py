import cv2
import numpy as np
import os


def detect_particle(video_path, threshold=0.12, region=(200, 0, 50, 380)):
    """
    Detects intensity change over time to identify particle passage in a channel.

    Parameters:
        video_path (str): Path to the video file.
        threshold (float): Threshold for intensity change to detect particle.
        region (tuple): (x, y, width, height) of the region of interest in the video frame.

    Returns:
        list of tuples: List of (frame_index, intensity_change) when a particle is detected.
    """
    # Create output folder if it doesn't exist
    output_folder = "detected_frames"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    particle_detections = []
    prev_intensity = None
    frame_index = 0
    x, y, w, h = region

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Crop to region of interest (ROI)
        roi = gray[y:y + h, x:x + w]

        # Calculate mean intensity in ROI
        mean_intensity = np.mean(roi)

        # Detect significant intensity change
        if prev_intensity is not None:
            intensity_change = abs(mean_intensity - prev_intensity)
            if intensity_change > threshold:
                # Calculate the time in seconds for this frame
                time_in_seconds = frame_index / frame_rate
                # Record detection with time and intensity change
                particle_detections.append((time_in_seconds, frame_index, intensity_change))

                # Save the detected frame
                save_path = os.path.join(output_folder, f"detected_frame_{frame_index}.png")
                cv2.imwrite(save_path, frame)
                print(f"Frame saved as {save_path}")

        # Update previous intensity
        prev_intensity = mean_intensity
        frame_index += 1

    cap.release()
    return particle_detections

def extract_frame(video_path, frame_number=0, save_path="extracted_frame.png"):
    """
    Extracts a specific frame from the video to calculate intensity and define ROI.

    Parameters:
        video_path (str): Path to the video file.
        frame_number (int): The index of the frame to extract (default is the first frame).

    Returns:
        numpy array: The extracted frame in grayscale.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    if not ret:
        print("Could not retrieve frame. Check video path or frame number.")
        return None

    # Convert frame to grayscale for intensity calculation
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cap.release()

    # Save the frame as an image file
    cv2.imwrite(save_path, gray_frame)
    print(f"Frame saved as {save_path}")

    cap.release()

    return gray_frame

# Usage example
video_path = "7um_PS_01001.mp4"

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames in video: {total_frames}")
cap.release()

detections = detect_particle(video_path)
print("Detected particles at times (seconds):", detections)
num_particles = len(detections)
print(f"Number of particles in video: {num_particles}")
# frame = extract_frame(video_path, frame_number=2813, save_path="extracted_frame.png")
# #
# # Calculate mean intensity of the entire frame as an example
# if frame is not None:
#     mean_intensity = np.mean(frame)
#     print("Mean Intensity of the Frame:", mean_intensity)