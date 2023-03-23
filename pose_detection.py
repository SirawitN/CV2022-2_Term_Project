from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates
from google.protobuf.json_format import MessageToDict
import cv2
import numpy as np
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def get_pose_detector(static_image_mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=False, smooth_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    pose_detector = mp.solutions.pose.Pose(
        static_image_mode=static_image_mode,
        smooth_landmarks=smooth_landmarks,
        enable_segmentation=enable_segmentation,
        smooth_segmentation=smooth_segmentation,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence)
    return pose_detector


def plot_pose_landmarks(frame: np.array, frame_num, pose_landmarks, frame_width, frame_height):
    landmarks = {}
    denormalized = {}

    for i in range(33):
        landmarks[f'frame_{frame_num}_{i}_x'] = 0
        landmarks[f'frame_{frame_num}_{i}_y'] = 0
        denormalized[f'frame_{frame_num}_{i}_x'] = 0
        denormalized[f'frame_{frame_num}_{i}_y'] = 0
    
    mp_drawing.draw_landmarks(
        frame,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    try:
        landmark_dict = MessageToDict(pose_landmarks)
    except:
        pass
    else:
        for i in range(len(landmark_dict['landmark'])):
            x, y = round(landmark_dict['landmark'][i]['x'], 3), round(landmark_dict['landmark'][i]['y'], 3)
            
            landmarks[f'frame_{frame_num}_{i}_x'] = x
            landmarks[f'frame_{frame_num}_{i}_y'] = y

            if x>1 or y>1:
                x = y = 0
            denormalized_coordinate = denormalize_coordinates(abs(x), abs(y), frame_width, frame_height)
            denormalized[f'frame_{frame_num}_{i}_x'] = denormalized_coordinate[0]
            denormalized[f'frame_{frame_num}_{i}_y'] = denormalized_coordinate[1]

    # print(landmarks, denormalized)

    return frame, landmarks, denormalized

header = ['index']
for i in range(33):
    header.append(f'landmark_{i}_x')
    header.append(f'landmark_{i}_y')

class MediapipePose:
    def __init__(self):
        self.pose = get_pose_detector()
        self.frame_num = 1
        self.landmarks = [header]
        self.denormalized_landmarks = [header]

    def process(self, frame: np.array):

        # convert the frame into RGB Colorspace
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # horizontally flip the frame
        frame = cv2.flip(frame, 1)

        frame.flags.writeable = False
        frame_h, frame_w, _ = frame.shape

        pose_results = self.pose.process(frame)

        # convert the colorspace back to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame.flags.writeable = True

        landmarks = None
        denormalized_landmarks = None

        if pose_results:
            frame, landmarks, denormalized_landmarks = plot_pose_landmarks(frame, self.frame_num, pose_results.pose_landmarks, frame_w, frame_h)
            self.frame_num += 1

            self.landmarks.append(list(landmarks.values()))
            self.denormalized_landmarks.append(list(denormalized_landmarks.values()))

        return frame
    
    def save(self, filename='test'):

        f = open(f'{filename}.csv', "w")
        for i in range(len(self.landmarks)):
            prefix = f'{i},' if i!=0 else ''
            f.write(f'{prefix}{",".join(str(l) for l in self.landmarks[i])}\n')
        f.close()
        
        f = open(f'{filename}_denormalized.csv', "w")
        for i in range(len(self.denormalized_landmarks)):
            prefix = f'{i},' if i!=0 else ''
            f.write(f'{prefix}{",".join(str(l) for l in self.denormalized_landmarks[i])}\n')
        f.close()

