import cv2
import mediapipe as mp
import moviepy.editor as mpedit
from glitch_this import ImageGlitcher
from PIL import Image
import numpy as np

# Initialize mediapipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to apply glitch effect
def apply_glitch_effect(frame):
    glitcher = ImageGlitcher()
    image = Image.fromarray(frame)
    glitched_image = glitcher.glitch_image(image, 2, color_offset=True)
    return np.array(glitched_image)

# Function to detect significant forward movement
def detect_left_side_movement(landmarks, threshold=0.12):
    nose = landmarks[0]
    left_leg = landmarks[25]
    right_leg = landmarks[26]
    leg_x = (left_leg.x + right_leg.x) / 2
    return nose.x < leg_x - threshold


# Load the video
input_video_path = "illuminati_1.mp4"
output_video_path = "output_video.mp4"
video = mpedit.VideoFileClip(input_video_path)

# Process each frame
frames = []
for frame in video.iter_frames():
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        if detect_left_side_movement(landmarks):
            frame = apply_glitch_effect(frame)
    
    frames.append(frame)

# Save the processed video
processed_video = mpedit.ImageSequenceClip(frames, fps=video.fps)
processed_video.write_videofile(output_video_path, codec='libx264')
