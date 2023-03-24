import cv2
import imutils, time
import mediapipe as mp
from imutils.video import VideoStream

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles

# img_in = '5.068.mp4'
img_in = 'P_Sade_144.mp4'
# img_in = 0

hands = mp_hands.Hands(min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

vs = VideoStream(src=img_in, framerate=30).start()
# cap = cv2.VideoCapture(img_in)

while True:
    frame = vs.read()

    if frame is not None:
        start_time = time.time()
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False

        hand_results = hands.process(frame)
        # pose_results = pose.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if hand_results.multi_hand_landmarks:
              for hand_landmarks in hand_results.multi_hand_landmarks:
                  print(hand_landmarks)
                  mp_drawing.draw_landmarks(
                      frame,
                      hand_landmarks,
                      mp_hands.HAND_CONNECTIONS)
        
        end_time = time.time()
        fps = int(1/(end_time-start_time))
        
        cv2.putText(frame, str(fps), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Processed frame', frame)


        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break


    else:
        break
    time.sleep(0.5)
# cleanup and closing frame
cv2.destroyAllWindows()
vs.stop()
