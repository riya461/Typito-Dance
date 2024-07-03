import cv2
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
    elif effect == 'red':
        frame[:, :, 2] = frame[:, :, 2] * intensity
    elif effect == 'green':
        frame[:, :, 1] = frame[:, :, 1] * intensity
    return frame

def rms_distance(landmarks1, landmarks2):
    total_distance = 0
    for l1, l2 in zip(landmarks1, landmarks2):
        total_distance += abs(((l1[0] - l2[0]) ** 2 + (l1[1] - l2[1]) ** 2) ** 0.5)
    # print(total_distance / len(landmarks1))
    return total_distance / len(landmarks1)
def forward_distance(landmarks1, landmarks2):
    # get the landmarks of the nose

    total_distance = 0
    l1 = landmarks1[0]
    l2 = landmarks2[0]
    # z 
    total_distance += abs(l1[2] - l2[2])
    return total_distance
def detect_movement(current_landmarks, prev_landmarks):
    movement_threshold = 0.02  # Threshold to consider as movement
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
            left_hand_landmarks = [pose_landmarks[i] for i in [12,14,16]]
            # right_hand_landmarks =[pose_landmarks[i] for i in [11,13,15]]
            left_leg_landmarks = [pose_landmarks[i] for i in[24,26,28]]
            # right_leg_landmarks = [pose_landmarks[i] for i in[23,25,27]]
            nose_landmark = [pose_landmarks[0]]

            # Detect movement
            current_landmarks = nose_landmark + left_hand_landmarks +  left_leg_landmarks #+ right_hand_landmarks + right_leg_landmarks 
            movement = detect_movement(current_landmarks, prev_landmarks)

            if prev_landmarks:
                dist = forward_distance(current_landmarks, prev_landmarks)

                print(dist)
                if dist > 0.5:
                    # pass
                    image = apply_color_effect(image, 'green')
                else:
                    left_hand_movement = detect_movement(left_hand_landmarks, prev_landmarks[1:3])
                    # right_hand_movement = detect_movement(right_hand_landmarks, prev_landmarks[2:4])
                    left_leg_movement = detect_movement(left_leg_landmarks, prev_landmarks[3:5])
                    # right_leg_movement = detect_movement(right_leg_landmarks, prev_landmarks[6:])


                    if left_hand_movement :#> right_hand_movement:
                        image = apply_color_effect(image, 'blue')
                    # elif right_hand_movement > left_hand_movement:
                    #     image = apply_color_effect(image, 'red')
                    # elif left_leg_movement :#> right_leg_movement:
                    #     image = apply_color_effect(image, 'blue')
                    # elif right_leg_movement > left_leg_movement:
                        # image = apply_color_effect(image, 'red')

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