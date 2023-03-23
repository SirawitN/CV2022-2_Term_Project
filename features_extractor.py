import cv2
import time
import imutils
from hand_detection import MediapipeHand
from pose_detection import MediapipePose
# from imutils.video import VideoStream

# img_in = '2566.0520.mp4'
img_in = 0

cap = cv2.VideoCapture(img_in)
hand_detector = MediapipeHand()
pose_detector = MediapipePose()

# used to record the time when we processed last frame
# prev_frame_time = 0

# # used to record the time at which we processed current frame
# new_frame_time = 0

# fps = cap.get(cv2.CAP_PROP_FPS)
# interval = int(1000/fps)

frames = 0
landmarks = []
denormalized_landmarks = []

if cap:
    while cap.isOpened():
        success, image = cap.read()
        if success:
            frames += 1
            start_time = time.time()

            # fps = str(int(1/(new_frame_time-prev_frame_time)))
            # fps = str(int(cap.get(cv2.CAP_PROP_FPS)))

            # prev_frame_time = new_frame_time
            # image = imutils.resize(image, height=240)
            
            # processed_frame, processed_landmarks, processed_denormalized = hand_detector.process(image)
            
            processed_frame, processed_landmarks, processed_denormalized = pose_detector.process(image)

            # if processed_landmarks is not None:
            #     landmarks.append(list(processed_landmarks.values()))
            # if processed_denormalized is not None:
            #     denormalized_landmarks.append(list(processed_denormalized.values()))

            end_time = time.time()

            fps = int(1/(end_time-start_time))

            # processed_frame = image.copy()

            # cv2.putText(image, str(fps), (5, 50),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Processed frame', processed_frame)

            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break
        else:
            # print(f'no video, Frames = {frames}, landmarks = {len(landmarks)}')
            # break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

    cap.release()
    cv2.destroyAllWindows()

else:
    print("[Error]: Cannot open a video with the given path. Please try again.")

