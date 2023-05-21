import cv2
import time
import os
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# video_input = './resources/5.068.mp4'
# video_input = 0
# cap = cv2.VideoCapture(video_input)


def get_holistic_model(static_image_mode=False, model_complexity=0, smooth_landmarks=True, enable_segmentation=False, refine_face_landmarks=False, min_detection_confidense=0.5, min_tracking_confidense=0.5):
    holistic_model = mp_holistic.Holistic(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        smooth_landmarks=smooth_landmarks,
        enable_segmentation=enable_segmentation,
        refine_face_landmarks=refine_face_landmarks,
        min_detection_confidence=min_detection_confidense,
        min_tracking_confidence=min_tracking_confidense)

    return holistic_model


def draw_landmarks(frame, results):
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(
        frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


# 151 headers
header = ['index']
# 33 pose landmarks with x and y values for each
for i in range(33):
    header.append(f'pose_{i}_x')
    header.append(f'pose_{i}_y')
# 21 hand landmarks with x and y values for each hand
for i in range(21):
    header.append(f'left_hand_{i}_x')
    header.append(f'left_hand_{i}_y')
for i in range(21):
    header.append(f'right_hand_{i}_x')
    header.append(f'right_hand_{i}_y')


class MediapipeHolistic:
    def __init__(self):
        self.holistic = get_holistic_model()
        self.frame_num = 0
        self.landmarks = [header]
        self.denormalized_landmarks = [header]

    def process(self, frame: np.array):
        start_time = time.time()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = self.holistic.process(frame)

        frame.flags.writeable = True
        fps = int(1/(time.time()-start_time))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(frame, str(fps), (5, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)
        draw_landmarks(frame, results)

        return frame, results

    def save_landmarks(self, frame_width, frame_height, results):
        if (not results.pose_landmarks and not results.left_hand_landmarks and not results.right_hand_landmarks):
            return
        self.frame_num += 1

        pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten(
        ) if results.pose_landmarks else np.zeros(33*2)
        left_hand = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]).flatten(
        ) if results.left_hand_landmarks else np.zeros(21*2)
        right_hand = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]).flatten(
        ) if results.right_hand_landmarks else np.zeros(21*2)

        cleaned_pose = [round(p, 3) if (p >= 0 and p <= 1)
                        else 0 for p in pose]
        cleaned_left = [round(p, 3) if (p >= 0 and p <= 1)
                        else 0 for p in left_hand]
        cleaned_right = [round(p, 3) if (p >= 0 and p <= 1)
                         else 0 for p in right_hand]

        denormalized_pose = []
        denormalized_left = []
        denormalized_right = []
        for i in range(0, len(cleaned_pose), 2):
            denormalized = denormalize_coordinates(abs(cleaned_pose[i]), abs(
                cleaned_pose[i+1]), frame_width, frame_height)
            denormalized_pose.append(denormalized[0])
            denormalized_pose.append(denormalized[1])
        for i in range(0, len(cleaned_left), 2):
            denormalized = denormalize_coordinates(abs(cleaned_left[i]), abs(
                cleaned_left[i+1]), frame_width, frame_height)
            denormalized_left.append(denormalized[0])
            denormalized_left.append(denormalized[1])
        for i in range(0, len(cleaned_right), 2):
            denormalized = denormalize_coordinates(abs(cleaned_right[i]), abs(
                cleaned_right[i+1]), frame_width, frame_height)
            denormalized_right.append(denormalized[0])
            denormalized_right.append(denormalized[1])

        self.landmarks.append(np.concatenate(
            ([self.frame_num], cleaned_pose, cleaned_left, cleaned_right)))
        self.denormalized_landmarks.append(np.concatenate(
            ([self.frame_num], denormalized_pose, denormalized_left, denormalized_right)))

    def export_to_csv(self, path='', filename='test'):

        save_dir = os.path.join(path, f'{filename}.csv')
        denormalized_save_dir = os.path.join(
            path, f'{filename}_denormalized.csv')

        f = open(save_dir, "w")
        for i in range(len(self.landmarks)):
            f.write(f'{",".join(str(l) for l in self.landmarks[i])}\n')
        f.close()

        f = open(denormalized_save_dir, "w")
        for i in range(len(self.denormalized_landmarks)):
            f.write(
                f'{",".join(str(l) for l in self.denormalized_landmarks[i])}\n')
        f.close()

    def clear_logs(self):
        self.frame_num = 0
        self.landmarks.clear()
        self.denormalized_landmarks.clear()
        self.landmarks = [header]
        self.denormalized_landmarks = [header]


# holistic = MediapipeHolistic()

# while cap.isOpened():
#     ret, frame = cap.read()

#     if ret:

#         frame, results = holistic.process(frame)
#         frame_h, frame_w, _ = frame.shape

#         holistic.save_landmarks(frame_w, frame_h, results)

#         cv2.imshow("Processed Frame", frame)
#         if (cv2.waitKey(1) & 0xFF == ord('q')):
#             break

#     else:
#         # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#         break

# holistic.export_to_csv()
# cap.release()
# cv2.destroyAllWindows()
