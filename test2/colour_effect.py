import cv2
import mediapipe as mp
from moviepy.editor import VideoFileClip

# Load the mediapipe models
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize global variables to store previous landmarks
prev_left_hand = None
prev_right_hand = None
prev_left_leg = None
prev_right_leg = None

def apply_color_effect(frame, effect):
    intensity = 0.5
    if effect == 'blue':
        frame[:, :, 0] = frame[:, :, 0] * intensity
    elif effect == 'green':
        frame[:, :, 1] = frame[:, :, 1] * intensity
    elif effect == 'red':
        frame[:, :, 2] = frame[:, :, 2] * intensity
    elif effect == 'yellow':
        frame[:, :, 0] = frame[:, :, 0] * intensity
        frame[:, :, 1] = frame[:, :, 1] * intensity
    return frame

def detect_movement(current_landmarks, prev_landmarks):
    movement_threshold = 0.02  # Threshold to consider as movement
    if prev_landmarks is None:
        return False

    for curr, prev in zip(current_landmarks, prev_landmarks):
        if abs(curr[0] - prev[0]) > movement_threshold or abs(curr[1] - prev[1]) > movement_threshold:
            return True
    return False

# Main function to process the video
def main():
    global prev_left_hand, prev_right_hand, prev_left_leg, prev_right_leg

    # Initialize mediapipe Pose model
    mp_holistic = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def process_frame(get_frame, t):
        global prev_left_hand, prev_right_hand, prev_left_leg, prev_right_leg

        frame = get_frame(t)
        frame_index = int(t * fps)

        # Resize frame
        frame = cv2.resize(frame, (new_width, new_height))

        # Recolor feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detections
        results = mp_holistic.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Check for pose landmarks
        if results.pose_landmarks:
            # Extract pose landmarks
            pose_landmarks = results.pose_landmarks.landmark

            # Extract specific landmarks for annotation
            left_hand_landmarks = [(pose_landmarks[i].x, pose_landmarks[i].y, pose_landmarks[i].z) for i in [14, 16]]
            right_hand_landmarks = [(pose_landmarks[i].x, pose_landmarks[i].y, pose_landmarks[i].z) for i in [13, 15]]
            left_leg_landmarks = [(pose_landmarks[i].x, pose_landmarks[i].y, pose_landmarks[i].z) for i in [26, 28]]
            right_leg_landmarks = [(pose_landmarks[i].x, pose_landmarks[i].y, pose_landmarks[i].z) for i in [25, 27]]

            # Detect movement and apply color effects
            if detect_movement(left_hand_landmarks, prev_left_hand):
                image = apply_color_effect(image, 'blue')
            if detect_movement(right_hand_landmarks, prev_right_hand):
                image = apply_color_effect(image, 'green')
            if detect_movement(left_leg_landmarks, prev_left_leg):
                image = apply_color_effect(image, 'red')
            if detect_movement(right_leg_landmarks, prev_right_leg):
                image = apply_color_effect(image, 'yellow')

            # Log landmarks
            landmarks = {
                'frame_index': frame_index,
                'left_hand': left_hand_landmarks,
                'right_hand': right_hand_landmarks,
                'left_leg': left_leg_landmarks,
                'right_leg': right_leg_landmarks
            }

            print(landmarks)  # Example of printing, adjust as needed

            # Update previous landmarks
            prev_left_hand = left_hand_landmarks
            prev_right_hand = right_hand_landmarks
            prev_left_leg = left_leg_landmarks
            prev_right_leg = right_leg_landmarks

            # Draw pose landmarks on the frame
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return image  # Return annotated frame

    video = 'hip_hop_1.mp4'
    out_video = 'hip_hop_1_final.mp4'
    scale_percent = 50
    clip = VideoFileClip(video)
    fps = clip.fps
    frame_width, frame_height = clip.size
    # Calculate new dimensions
    new_width = int(frame_width * scale_percent / 100)
    new_height = int(frame_height * scale_percent / 100)

    # Process video frames
    processed_clip = clip.fl(process_frame, apply_to=['video'])

    # Save the processed video
    processed_clip.write_videofile(out_video, codec='libx264')

    # Release the holistic model
    mp_holistic.close()

if __name__ == '__main__':
    main()
