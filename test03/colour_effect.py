import cv2
import mediapipe as mp
from moviepy.editor import VideoFileClip

# Load the mediapipe models
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Store previous landmarks
prev_landmarks = None
movement_distances = []

def zoom_and_pan(frame, effect):
    h, w, _ = frame.shape
    zoom_factor = 0.1  # Adjust this value to control zoom level
    pan_factor = 0.1   # Adjust this value to control panning distance

    # Calculate cropping dimensions
    crop_w = int(w * (1 - zoom_factor))
    crop_h = int(h * (1 - zoom_factor))

    if effect == 'left':  # Pan left
        start_x = int(w * pan_factor)
    elif effect == 'right':  # Pan right
        start_x = int(w * (1 - pan_factor - (1 - zoom_factor)))
    else:
        start_x = (w - crop_w) // 2  # Center

    start_y = (h - crop_h) // 2  # Center vertically

    # Crop the frame
    cropped_frame = frame[start_y:start_y + crop_h, start_x:start_x + crop_w]

    # Resize back to original dimensions
    zoomed_frame = cv2.resize(cropped_frame, (w, h))

    return zoomed_frame

def distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def detect_movement(current_landmarks, prev_landmarks):
    movement_threshold = 0.3  # Threshold to consider as movement
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

def detect_pan_direction(left_landmarks, right_landmarks, prev_left_landmarks, prev_right_landmarks):
    left_movement = detect_movement(left_landmarks, prev_left_landmarks)
    right_movement = detect_movement(right_landmarks, prev_right_landmarks)
    
    if left_movement > right_movement:
        return 'left'
    elif right_movement > left_movement:
        return 'right'
    else:
        return None

def main():
    global prev_landmarks

    # Initialize mediapipe Pose model
    mp_holistic = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

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

            if prev_landmarks:
                # Detect pan direction
                pan_direction = detect_pan_direction(
                    left_hand_landmarks + left_leg_landmarks, 
                    right_hand_landmarks + right_leg_landmarks, 
                    prev_landmarks[:len(left_hand_landmarks) + len(left_leg_landmarks)], 
                    prev_landmarks[len(left_hand_landmarks) + len(left_leg_landmarks):]
                )

                if pan_direction:
                    image = zoom_and_pan(image, pan_direction)

            # Update previous landmarks
            prev_landmarks = left_hand_landmarks + right_hand_landmarks + left_leg_landmarks + right_leg_landmarks

            # Draw pose landmarks on the frame
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return image, t  # Return annotated frame and timestamp

    video = '../inputs/illuminati_1.mp4'
    out_video = 'illuminati_1_final000.mp4'
    scale_percent = 50
    clip = VideoFileClip(video)
    fps = clip.fps
    frame_width, frame_height = clip.size
    # Calculate new dimensions
    new_width = int(frame_width * scale_percent / 100)
    new_height = int(frame_height * scale_percent / 100)

    def process_clip(get_frame, t):
        image, timestamp = process_frame(get_frame, t)
        return image

    processed_clip = clip.fl(process_clip, apply_to=['video'])

    # Save the processed video
    processed_clip.write_videofile(out_video, codec='libx264', fps=fps)

    # Release the holistic model
    mp_holistic.close()

if __name__ == '__main__':
    main()


# def apply_color_effect(frame, effect):
#     intensity = 0.5
#     if effect == 'blue':
#         frame[:, :, 0] = frame[:, :, 0] * intensity
#     elif effect == 'green':
#         frame[:, :, 1] = frame[:, :, 1] * intensity
#     elif effect == 'red'import cv2
import mediapipe as mp
from moviepy.editor import VideoFileClip

# Load the mediapipe models
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Store previous landmarks
prev_landmarks = None

def apply_color_effect(frame, effect):
    intensity = 0.5
    if effect == 'blue':
        frame[:, :, 0] = frame[:, :, 0] * intensity
    elif effect == 'green':
        frame[:, :, 1] = frame[:, :, 1] * intensity
    elif effect == 'red':
        frame[:, :, 2] = frame[:, :, 2] * intensity
    return frame

def rms_distance(landmarks1, landmarks2):
    total_distance = 0
    for l1, l2 in zip(landmarks1, landmarks2):
        total_distance += ((l1[0] - l2[0]) ** 2 + (l1[1] - l2[1]) ** 2) ** 0.5
    return total_distance / len(landmarks1)

def detect_movement(current_landmarks, prev_landmarks):
    movement_threshold = 0.1  # Threshold to consider as movement
    if prev_landmarks is None:
        return 0

    rms_dist = rms_distance(current_landmarks, prev_landmarks)
    if rms_dist > movement_threshold:
        return rms_dist
    else:
        return 0

def main():
    global prev_landmarks

    # Initialize mediapipe Pose model
    mp_holistic = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

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
        results = mp_holistic.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Check for pose landmarks
        if results.pose_landmarks:
            # Extract pose landmarks
            pose_landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]

            # Separate landmarks for hands and legs
            left_hand_landmarks = pose_landmarks[15:17]
            right_hand_landmarks = pose_landmarks[16:18]
            left_leg_landmarks = pose_landmarks[25:27]
            right_leg_landmarks = pose_landmarks[26:28]

            # Detect movement
            current_landmarks = left_hand_landmarks + right_hand_landmarks
            movement = detect_movement(current_landmarks, prev_landmarks)

            if movement > 0.1:  # Adjust threshold as needed
                left_movement = detect_movement(left_hand_landmarks, prev_landmarks[:2])
                right_movement = detect_movement(right_hand_landmarks, prev_landmarks[2:4])

                if left_movement > right_movement:
                    image = apply_color_effect(image, 'blue')
                elif right_movement > left_movement:
                    image = apply_color_effect(image, 'red')
                else:
                    image = apply_color_effect(image, 'green')

            # Update previous landmarks
            prev_landmarks = current_landmarks

            # Draw pose landmarks on the frame
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return image, t  # Return annotated frame and timestamp

    video = '../inputs/illuminati_1.mp4'
    out_video = 'illuminati_1_final.mp4'
    scale_percent = 50
    clip = VideoFileClip(video)
    fps = clip.fps
    frame_width, frame_height = clip.size
    # Calculate new dimensions
    new_width = int(frame_width * scale_percent / 100)
    new_height = int(frame_height * scale_percent / 100)

    def process_clip(get_frame, t):
        image, timestamp = process_frame(get_frame, t)
        return image

    processed_clip = clip.fl(process_clip, apply_to=['video'])

    # Save the processed video
    processed_clip.write_videofile(out_video, codec='libx264', fps=fps)

    # Release the holistic model
    mp_holistic.close()

if __name__ == '__main__':
    main()

#         frame[:, :, 2] = frame[:, :, 2] * intensity
#     elif effect == 'yellow':
#         frame[:, :, 0] = frame[:, :, 0] * intensity
#         frame[:, :, 1] = frame[:, :, 1] * intensity
#     return frame