import cv2
import mediapipe as mp
from moviepy.editor import VideoFileClip, ImageSequenceClip, CompositeVideoClip, TextClip
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

# Function to zoom in on the frame
def zoom_in(frame, zoom_factor):
    h, w, _ = frame.shape
    zoom_h = int(h * zoom_factor)
    zoom_w = int(w * zoom_factor)
    zoomed_frame = cv2.resize(frame, (zoom_w, zoom_h))
    return zoomed_frame

# Function to zoom out from the frame
def zoom_out(frame, zoom_factor):
    h, w, _ = frame.shape
    zoom_h = int(h / zoom_factor)
    zoom_w = int(w / zoom_factor)
    zoomed_frame = cv2.resize(frame, (zoom_w, zoom_h))
    return zoomed_frame

# Define the time intervals for applying effects (in seconds)
effect_intervals = {
    'zoom_in': [(0, 3), (10, 12)],
    'zoom_out': [(3, 5), (12, 15)],
    # 'glitch': [(3, 5), (10, 12)]
}

# Load the video
input_video_path = "illuminati_1.mp4"
output_video_path = "output_video_with_effects.mp4"
video = VideoFileClip(input_video_path)

# Function to check if the current time is within the effect intervals
def is_in_effect_interval(current_time, intervals):
    return any(start <= current_time <= end for start, end in intervals)

# Process each frame
processed_frames = []
for current_time in np.arange(0, video.duration, 1 / video.fps):
    frame = video.get_frame(current_time)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Apply effects based on intervals
    if is_in_effect_interval(current_time, effect_intervals['zoom_in']):
        frame = zoom_in(frame, 1.2)  # Example zoom factor of 1.2
    
    if is_in_effect_interval(current_time, effect_intervals['zoom_out']):
        frame = zoom_out(frame, 1.2)  # Example zoom factor of 1.2

    # if is_in_effect_interval(current_time, effect_intervals['glitch']):
    #     frame = apply_glitch_effect(frame)
    
    processed_frames.append(frame)

# Ensure all frames are resized to the same dimensions
frame_size = processed_frames[0].shape[:2]
processed_frames = [cv2.resize(frame, (frame_size[1], frame_size[0])) for frame in processed_frames]

# Convert frames to ImageSequenceClip
processed_video = ImageSequenceClip(processed_frames, fps=video.fps)

# Extract audio from the original video
audio_clip = video.audio

# Set audio for the processed video
final_video = processed_video.set_audio(audio_clip)

# Write the final video with effects and audio
final_video.write_videofile(output_video_path, codec='libx264')
