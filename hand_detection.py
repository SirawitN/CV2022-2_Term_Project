from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates
from google.protobuf.json_format import MessageToDict
import cv2, os
import numpy as np
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def get_hand_detector(static_image_mode=False, max_num_hands=2, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    hand_detector = mp.solutions.hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence)
    return hand_detector


def plot_hand_landmarks(frame: np.array, frame_num, hand_landmarks, frame_width, frame_height):
    landmarks = {}
    denormalized = {}

    for i in range(1, 22):
        landmarks[f'frame_{frame_num}_left_{i}_x'] = 0
        landmarks[f'frame_{frame_num}_left_{i}_y'] = 0
        denormalized[f'frame_{frame_num}_left_{i}_x'] = 0
        denormalized[f'frame_{frame_num}_left_{i}_y'] = 0
    for i in range(1, 22):
        landmarks[f'frame_{frame_num}_right_{i}_x'] = 0
        landmarks[f'frame_{frame_num}_right_{i}_y'] = 0
        denormalized[f'frame_{frame_num}_right_{i}_x'] = 0
        denormalized[f'frame_{frame_num}_right_{i}_y'] = 0

    for landmark in hand_landmarks:
        landmark_dict = MessageToDict(landmark)
        side = 'left' if landmark_dict['landmark'][0]['x'] < 0.5 else 'right'

        for j in range(1, 22):
            x, y = round(landmark_dict['landmark'][j-1]['x'],
                         3), round(landmark_dict['landmark'][j-1]['y'], 3)
            landmarks[f'frame_{frame_num}_{side}_{j}_x'] = x
            landmarks[f'frame_{frame_num}_{side}_{j}_y'] = y

            if x>1 or y>1:
                x = y = 0

            denormalized_coordinate = denormalize_coordinates(
                abs(x), abs(y), frame_width, frame_height)
            
            # print(x, y, denormalized_coordinate)
            denormalized[f'frame_{frame_num}_{side}_{j}_x'] = denormalized_coordinate[0]
            denormalized[f'frame_{frame_num}_{side}_{j}_y'] = denormalized_coordinate[1]

        mp_drawing.draw_landmarks(frame, landmark, mp_hands.HAND_CONNECTIONS)

    return frame, landmarks, denormalized

header = ['index']
for i in range(21):
    header.append(f'landmark_left_{i}_x')
    header.append(f'landmark_left_{i}_y')
for i in range(21):
    header.append(f'landmark_right_{i}_x')
    header.append(f'landmark_right_{i}_y')


class MediapipeHand:

    def __init__(self):
        self.hand = get_hand_detector()
        self.frame_num = 1
        self.landmarks = [header]
        self.denormalized_landmarks = [header]

    def process(self, frame: np.array):

        # convert the frame into RGB Colorspace
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # horizontally flip the frame
        # frame = cv2.flip(frame, 1)

        frame.flags.writeable = False
        frame_h, frame_w, _ = frame.shape

        # print(frame_h, frame_w)

        hand_results = self.hand.process(frame)

        # convert the colorspace back to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame.flags.writeable = True

        landmarks = None
        denormalized_landmarks = None

        if hand_results.multi_hand_landmarks:
            # print(hand_results.multi_handedness)
            frame, landmarks, denormalized_landmarks = plot_hand_landmarks(
                frame, self.frame_num, hand_results.multi_hand_landmarks, frame_w, frame_h)
            self.frame_num += 1

            self.landmarks.append(list(landmarks.values()))
            self.denormalized_landmarks.append(list(denormalized_landmarks.values()))

        # if hand_results.multi_handedness:
        #     # for handedness in hand_results.multi_handedness:
        #     for idx, hand_handedness in enumerate(hand_results.multi_handedness):
        #         handedness_dict = MessageToDict(hand_handedness)
        #         handedness = handedness_dict['classification'][0]
        #         # print(hand_results.multi_hand_world_landmarks[idx])
        #         # print(handedness['index'], handedness['score'])
        #         print(len(hand_results.multi_hand_landmarks))

        return frame
    
    def save(self, path='', filename='test'):

        save_dir = os.path.join(path, f'{filename}.csv')
        denormalized_save_dir = os.path.join(path, f'{filename}_denormalized.csv')
        f = open(save_dir, "w")
        for i in range(len(self.landmarks)):
            prefix = f'{i},' if i!=0 else ''
            f.write(f'{prefix}{",".join(str(l) for l in self.landmarks[i])}\n')
        f.close()
        
        f = open(denormalized_save_dir, "w")
        for i in range(len(self.denormalized_landmarks)):
            prefix = f'{i},' if i!=0 else ''
            f.write(f'{prefix}{",".join(str(l) for l in self.denormalized_landmarks[i])}\n')
        f.close()

    def clear_logs(self):
        self.landmarks.clear()
        self.denormalized_landmarks.clear()
        self.landmarks = [header]
        self.denormalized_landmarks = [header]
