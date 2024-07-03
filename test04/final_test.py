import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from moviepy.editor import ImageSequenceClip

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def flash(frame):
    frame[:, :, 0] = np.clip(frame[:, :, 0] * 1.5, 0, 255)
    frame[:, :, 1] = np.clip(frame[:, :, 1] * 1.5, 0, 255)
    frame[:, :, 2] = np.clip(frame[:, :, 2] * 1.5, 0, 255)
    return frame

video_path = '../inputs/illuminati_1.mp4'
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)

frame_numbers = []
timestamps = []
forward_movement_z = []
left_hand_distances = []
left_leg_distances = []
right_hand_distances = []
right_leg_distances = []

frame_number = 0

# First pass to collect data
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate the timestamp
    timestamp = frame_number / fps

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        frame_numbers.append(frame_number)
        timestamps.append(timestamp)

        forward_movement_z.append(landmarks[0].z)

        left_hand_distance = np.sqrt((landmarks[12].x - landmarks[14].x)**2 + (landmarks[12].y - landmarks[14].y)**2) + \
                             np.sqrt((landmarks[14].x - landmarks[16].x)**2 + (landmarks[14].y - landmarks[16].y)**2)
        
        left_leg_distance = np.sqrt((landmarks[24].x - landmarks[26].x)**2 + (landmarks[24].y - landmarks[26].y)**2) + \
                            np.sqrt((landmarks[26].x - landmarks[28].x)**2 + (landmarks[26].y - landmarks[28].y)**2)
        
        right_hand_distance = np.sqrt((landmarks[11].x - landmarks[13].x)**2 + (landmarks[11].y - landmarks[13].y)**2) + \
                              np.sqrt((landmarks[13].x - landmarks[15].x)**2 + (landmarks[13].y - landmarks[15].y)**2)
        
        right_leg_distance = np.sqrt((landmarks[23].x - landmarks[25].x)**2 + (landmarks[23].y - landmarks[25].y)**2) + \
                             np.sqrt((landmarks[25].x - landmarks[27].x)**2 + (landmarks[25].y - landmarks[27].y)**2)

        left_hand_distances.append(left_hand_distance)
        left_leg_distances.append(left_leg_distance)
        right_hand_distances.append(right_hand_distance)
        right_leg_distances.append(right_leg_distance)

    frame_number += 1

cap.release()

# Find the troughs in the data
troughs_left_hand, _ = find_peaks(np.array(left_hand_distances))
troughs_right_hand, _ = find_peaks(np.array(right_hand_distances))

# Combine the frame numbers of the troughs
trough_frames = set(troughs_left_hand).union(set(troughs_right_hand))
#  Sort the peaks by their distances in descending order
top_peaks_left_hand = sorted(troughs_left_hand, key=lambda x: left_hand_distances[x], reverse=True)[:5]
top_peaks_right_hand = sorted(troughs_right_hand, key=lambda x: right_hand_distances[x], reverse=True)[:5]

# Combine the frame numbers of the top peaks
top_peak_frames = set(top_peaks_left_hand).union(set(top_peaks_right_hand))

# Print the top 5 peaks and their distances
print("Top 5 Peaks in Left Hand Distances:", [(frame_numbers[i], left_hand_distances[i]) for i in top_peaks_left_hand])
print("Top 5 Peaks in Right Hand Distances:", [(frame_numbers[i], right_hand_distances[i]) for i in top_peaks_right_hand])
# Second pass to apply flash effect based on trough frames
cap = cv2.VideoCapture(video_path)
frame_number = 0
frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_number in top_peak_frames:
        frame = flash(frame)

    

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame_rgb)
    frame_number += 1

# Create a video clip from the frames
clip = ImageSequenceClip(frames, fps=fps)

# Write the video file
output_path = 'output_with_flash2.mp4'
clip.write_videofile(output_path, codec='libx264')
cap.release()
cv2.destroyAllWindows()

plt.figure(figsize=(15, 10))

plt.subplot(4, 1, 1)
plt.plot(timestamps, forward_movement_z, label='Forward Movement (z)', color='blue')
plt.xlabel('Timestamp (s)')
plt.ylabel('Forward Movement (z)')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(timestamps, left_hand_distances, label='Left Hand Distance', color='green')
plt.plot(timestamps, right_hand_distances, label='Right Hand Distance', color='red')
plt.xlabel('Timestamp (s)')
plt.ylabel('Hand Distance')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(timestamps, left_leg_distances, label='Left Leg Distance', color='purple')
plt.plot(timestamps, right_leg_distances, label='Right Leg Distance', color='orange')
plt.xlabel('Timestamp (s)')
plt.ylabel('Leg Distance')
plt.legend()

# Save the plot
plt.savefig('plot.png')

# Print the troughs
print("Troughs in Left Hand Distances:", troughs_left_hand)
print("Troughs in Right Hand Distances:", troughs_right_hand)
print("Min left hand distance:", min(left_hand_distances))
print("Min right hand distance:", min(right_hand_distances))
