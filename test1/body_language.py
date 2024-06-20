import cv2
import csv
import os
import mediapipe as mp
import numpy as np
from PIL import Image
from glitch_this import ImageGlitcher
from moviepy.editor import VideoFileClip

def apply_glitch_this(image):
    glitcher = ImageGlitcher()
    pil_image = Image.fromarray(image)
    glitched_image = glitcher.glitch_image(pil_image, 4, color_offset=True)
    return np.array(glitched_image)

def calculate_rms_distances(prev_landmarks, curr_landmarks):
    distances = []
    for i in range(len(prev_landmarks)):
        dist = np.linalg.norm(np.array(prev_landmarks[i][:3]) - np.array(curr_landmarks[i][:3]))
        distances.append(dist)
    return np.sqrt(np.mean(np.square(distances)))

def calculate_distances(prev_landmarks, curr_landmarks):
    distances = []
    for i in range(len(prev_landmarks)):
        dist = np.linalg.norm(np.array(prev_landmarks[i][:3]) - np.array(curr_landmarks[i][:3]))
        distances.append(dist)
    return np.mean(distances)

def apply_sepia(frame):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_frame = cv2.transform(frame, sepia_filter)
    sepia_frame = np.clip(sepia_frame, 0, 255).astype(np.uint8)
    return sepia_frame

# File paths
video = '../inputs/illuminati_1.mp4'
val = video.split('.')[-2].split('/')[-1]
print(val)
csv_file = f'../output_1/{val}.csv'
final_video = f'../output_1/{val}_final.mp4'

# Initialize Mediapipe holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Define the scale for resizing
scale_percent = 50  # Resize to 50% of the original size

# Open CSV file for writing
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header row 
    header = ['frame'] + [f'{landmark}_{coord}' for landmark in range(33) for coord in ('x', 'y', 'z', 'visibility')]
    writer.writerow(header)

    prev_landmarks = None  # Initialize previous landmarks

    # Initialize holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        def process_frame(get_frame, t):
            global prev_landmarks 
            frame = get_frame(t)
            frame_index = int(t * fps)

            # Resize frame
            frame = cv2.resize(frame, (new_width, new_height))

            # Recolor feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detections
            results = holistic.process(image)

            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Check for pose landmarks
            if results.pose_landmarks:
                # Extract pose landmarks
                pose_landmarks = results.pose_landmarks.landmark[11:33]
                curr_landmarks = [(landmark.x, landmark.y, landmark.z, landmark.visibility) for landmark in pose_landmarks]
                if prev_landmarks is not None:
                    distance = calculate_distances(prev_landmarks, curr_landmarks)
                    
                    if distance > 0.25:  # Threshold
                        # Apply effect
                        frame = apply_glitch_this(frame)
                
                landmarks = [frame_index] + [coord for landmark in curr_landmarks for coord in landmark]
                writer.writerow(landmarks)

                # Draw pose landmarks on the frame
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                
                prev_landmarks = curr_landmarks

            return frame

        # Open video file with moviepy
        clip = VideoFileClip(video)
        fps = clip.fps
        frame_width, frame_height = clip.size

        # Calculate new dimensions
        new_width = int(frame_width * scale_percent / 100)
        new_height = int(frame_height * scale_percent / 100)

        # Process video frames
        processed_clip = clip.fl(process_frame)

        # Save the processed video
        processed_clip.write_videofile(final_video, codec='libx264')

print("Final video saved as final.mp4")
