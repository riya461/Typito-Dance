import cv2
import mediapipe as mp
from moviepy.editor import VideoFileClip

# Load the mediapipe models
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Store previous landmarks
prev_left_hand = None
prev_right_hand = None
prev_left_leg = None
prev_right_leg = None

def zoom_and_pan(frame, effect):
    h, w, _ = frame.shape
    zoom_factor = 0.1  # Adjust this value to control zoom level
    pan_factor = 0.1   # Adjust this value to control panning distance

    # Calculate cropping dimensions
    crop_w = int(w * (1 - zoom_factor))
    crop_h = int(h * (1 - zoom_factor))

    if effect in ['blue', 'red']:  # Pan left
        start_x = int(w * pan_factor)
    elif effect in ['green', 'yellow']:  # Pan right
        start_x = int(w * (1 - pan_factor - (1 - zoom_factor)))
    else:
        start_x = (w - crop_w) // 2  # Center

    start_y = (h - crop_h) // 2  # Center vertically

    # Crop the frame
    cropped_frame = frame[start_y:start_y + crop_h, start_x:start_x + crop_w]

    # Resize back to original dimensions
    zoomed_frame = cv2.resize(cropped_frame, (w, h))

    return zoomed_frame

def apply_color_effect(frame, effect):
    intensity = 0.5
    if effect == 'blue':
        # left hand 
        frame[:, :, 0] = frame[:, :, 0] * intensity
    elif effect == 'green':
        # right hand
        frame[:, :, 1] = frame[:, :, 1] * intensity
    elif effect == 'red':
        # left leg
        frame[:, :, 2] = frame[:, :, 2] * intensity
    elif effect == 'yellow':
        # right leg
        frame[:, :, 0] = frame[:, :, 0] * intensity
        frame[:, :, 1] = frame[:, :, 1] * intensity
    return frame

def distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def detect_movement(current_landmarks, prev_landmarks):
    movement_threshold = 0.05  # Threshold to consider as movement
    if prev_landmarks is None:
        return 0

    total_distance = 0
    for curr, prev in zip(current_landmarks, prev_landmarks):
        total_distance += distance(curr[0], curr[1], prev[0], prev[1])

    average_distance = total_distance / len(current_landmarks)
    if average_distance > movement_threshold:
        return average_distance
    else:
        return 0

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
            left_hand_landmarks = [(pose_landmarks[i].x, pose_landmarks[i].y, pose_landmarks[i].z) for i in [15, 17]]
            right_hand_landmarks = [(pose_landmarks[i].x, pose_landmarks[i].y, pose_landmarks[i].z) for i in [16, 18]]
            left_leg_landmarks = [(pose_landmarks[i].x, pose_landmarks[i].y, pose_landmarks[i].z) for i in [25, 27]]
            right_leg_landmarks = [(pose_landmarks[i].x, pose_landmarks[i].y, pose_landmarks[i].z) for i in [26, 28]]

            # Detect movement and apply color effects
            left_hand_movement = detect_movement(left_hand_landmarks, prev_left_hand)
            right_hand_movement = detect_movement(right_hand_landmarks, prev_right_hand)
            left_leg_movement = detect_movement(left_leg_landmarks, prev_left_leg)
            right_leg_movement = detect_movement(right_leg_landmarks, prev_right_leg)

            movement_array = [left_hand_movement, right_hand_movement, left_leg_movement, right_leg_movement]

            # Apply color effects
            max_movement = max(movement_array)
            index = movement_array.index(max_movement)

            if max_movement > 0:
                effects = ['blue', 'green', 'red', 'yellow']
                effect = effects[index]
                image = zoom_and_pan(image, effect)
                # image = apply_color_effect(image, effect)

            # Update previous landmarks
            prev_left_hand = left_hand_landmarks
            prev_right_hand = right_hand_landmarks
            prev_left_leg = left_leg_landmarks
            prev_right_leg = right_leg_landmarks

            # Draw pose landmarks on the frame
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return image  # Return annotated frame

    video = 'illuminati_1.mp4'
    out_video = 'illuminati_1_final.mp4'
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
    processed_clip.write_videofile(out_video, codec='libx264', fps=50)

    # Release the holistic model
    mp_holistic.close()

if __name__ == '__main__':
    main()
