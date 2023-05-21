import cv2
import imutils
import os, time
import numpy as np
from mediapipe_holistic import MediapipeHolistic
# from imutils.video import VideoStream

# Instraction: How to use this code
# 1. Install all necessary modules, meidapipe, opencv
# 2. Download videos from https://drive.google.com/drive/folders/1SSdPAg4gxv2S0PP7rEGp7awuVHKbTwO8?usp=sharing
# 3. Create a 'resources' directory
# 4. Extract the downloaded file and rename it to 'CompVisionDataset1'
# 5. Put the 'CompVisionDataset1' in the 'resources' directory
# 6. Finish

# Examples of detected landmarks are in test.csv and test_denormalized.csv

resource_dir = os.path.join('resources', 'CompVisionDataset')

save_dir = 'results'
holistic_detector = MediapipeHolistic()
    
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

stop = False

# find each video in the class folder
files = os.listdir(resource_dir)
files = [f for f in files if os.path.isfile(resource_dir+'/'+f)]

for file in files:

    file_path = os.path.join(resource_dir, file)
    
    if(os.path.exists(file_path)):
        cap = cv2.VideoCapture(file_path)
        if cap:
            while cap.isOpened():
                success, image = cap.read()
                if success:
                    # image = imutils.resize(image, height=240)
                
                    processed_frame, results = holistic_detector.process(image)
                    frame_h, frame_w, _ = image.shape
                    holistic_detector.save_landmarks(frame_w, frame_h, results)

                    cv2.imshow('Processed frame', processed_frame)
                    if (cv2.waitKey(1) & 0xFF == ord('q')):
                        stop = True;
                        break
                else:
                    break


            cap.release()
            cv2.destroyAllWindows()

        else:
            print("[Error]: Cannot open a video with the given path. Please try again.")

        if stop:
            break
        else:
            holistic_detector.export_to_csv(save_dir, file[:-4])
            holistic_detector.clear_logs()
        
    else:
        raise Exception(f'An error occured, the {file_path} doesn\'t exist.')


