import sys
import random
import string

import torch
import numpy as np
import pandas as pd

import cv2
from feature_extractor.mediapipe_holistic import MediapipeHolistic

from lstm import SimpleLSTM

import time

# constant varaible
INPUT_WIDTH = 1280
INPUT_HEIGHT = 720
HAND_INDEXS = np.array([0,1,4,5,8,9,12,13,16,17,20]) # get just the tip of finger
# POSE_INDEXS = np.array([0,11, 12, 13, 14, 15, 16, 23, 24]) # get just the upper body of pose estimated
POSE_INDEXS = np.array([0,11, 12]) # get just the upper body of pose estimated
HAND_COLUMN_INDEXES = np.array([66, 67, 68, 69, 74, 75, 76, 77, 82, 83, 84, 85, 90, 91, 92, 93, 98, 99, 100, 101, 106, 107, 108, 109, 110, 111, 116, 117, 118, 119, 124, 125, 126, 127, 132, 133, 134, 135, 140, 141, 142, 143, 148, 149])

col_name = []
hand_col_name = []
for pidx in POSE_INDEXS:
    col_name.append('pose_' + str(pidx) + '_x')
    col_name.append('pose_' + str(pidx) + '_y')
for hidx in HAND_INDEXS:
    col_name.append('left_hand_' + str(hidx) + '_x')
    col_name.append('left_hand_' + str(hidx) + '_y')
    hand_col_name.append('left_hand_' + str(hidx) + '_x')
    hand_col_name.append('left_hand_' + str(hidx) + '_y')
for hidx in HAND_INDEXS:
    col_name.append('right_hand_' + str(hidx) + '_x')
    col_name.append('right_hand_' + str(hidx) + '_y')
    hand_col_name.append('right_hand_' + str(hidx) + '_x')
    hand_col_name.append('right_hand_' + str(hidx) + '_y')

def get_random_string(length):
    # With combination of lower and upper case
    result_str = ''.join(random.choice(string.ascii_letters) for i in range(length))
    # print random string
    return result_str

def predict(model, x):
    y_pred = model(x)
    return y_pred.argmax(dim=1).tolist()

def set_res(cap, x,y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
    return str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def preprocess_data(df):
    data = df.copy()
    cols = df.columns
    
    col_name_not_none = hand_col_name
    
    # remove null values rows
    not_null_row = []
    for i in range(len(data)):
        if (data.iloc[i][col_name_not_none] > 0).sum() > 0:
            not_null_row.append(i)
    
    data = data.iloc[not_null_row]
    
    x_col = [col for col in cols if 'x' in col]
    y_col = [col for col in cols if 'y' in col]
    
    data[x_col] = data[x_col].replace(0, 1.5)
    data[y_col] = data[y_col].replace(0, 1.5)
    
    # assume that we have pose_0
    mid_x = data['pose_0_x']
    mid_y = data['pose_0_y']
    shoulder_left_x = data['pose_11_x']
    shoulder_left_y = data['pose_11_y']
    shoulder_right_x = data['pose_12_x']
    shoulder_right_y = data['pose_12_y']
    
    shoulder_diff = np.sqrt((shoulder_left_x - shoulder_right_x)**2 + (shoulder_left_y - shoulder_right_y)**2)
    
    data[x_col] = data[x_col].apply(lambda x: x - mid_x)
    data[y_col] = data[y_col].apply(lambda x: x - mid_y)
        
    for col in cols:
        data[col] = data[col].values / shoulder_diff.values
    
    return data

def sample_data(df, len_sample=30):
    sample_idx = np.linspace(0, df.shape[0]-1, len_sample, dtype=int)
    return df.iloc[sample_idx].to_numpy()

def is_detecting_hand(landmarks_frame):
    landmarks_np = np.array(landmarks_frame[1:])
    if (landmarks_np[HAND_COLUMN_INDEXES] > 0).sum() > 0:
        return True
    return False

# def get_prediction(raw_df):
    

if __name__ == "__main__":
    
    mapping_file = sys.path[0] + '/../resources/Class_mapping_start_with_0.txt'
    ctol = {}
    ltoc = {}
    with open(mapping_file, encoding='utf-8') as f:
        for line in f:
            (key, val) = line.split()
            ctol[int(key)] = val
            ltoc[val] = int(key)
    
    model = SimpleLSTM(input_dim=len(col_name), hidden_dim=256, classes=len(ctol), leaky_relu=True)
    model.load_state_dict(torch.load(sys.path[0] + '/../checkpoint/model_epoch_106_with_leaky_relu.pt'))
    model.to('cuda')
    
    inputStream = cv2.VideoCapture(0)
    processed_frames = []
    set_res(inputStream, INPUT_WIDTH, INPUT_HEIGHT)
    holistic_detector = MediapipeHolistic()
    
    isDetecting = False
    detecting_frame_count = 0
    not_detecting_frame_count = 0
    
    try:
        while True:
            success, videoFrame = inputStream.read()
            # videoFrame empty case handler
            if success:
                processed_frame, results = holistic_detector.process(videoFrame)
                holistic_detector.save_landmarks(INPUT_WIDTH, INPUT_HEIGHT, results)
                
                last_frame = holistic_detector.landmarks[-1]
                
                if is_detecting_hand(last_frame):
                    detecting_frame_count += 1
                    not_detecting_frame_count = 0
                else:
                    detecting_frame_count = 0
                    not_detecting_frame_count += 1
                
                if detecting_frame_count > 5:
                    isDetecting = True
                if not_detecting_frame_count > 10:
                    if isDetecting:
                        print("-------------trigger detection-------------")
                        data_df = pd.DataFrame(data=holistic_detector.landmarks[1:], columns=holistic_detector.landmarks[0])
                        holistic_detector.clear_logs()
                        
                        # preprocessed data
                        data_df = data_df.drop(['index'], axis=1)
                        data_df = data_df[col_name]
                        processed_data = preprocess_data(data_df)
                        inputs = sample_data(processed_data)
                        inputs = np.expand_dims(inputs, axis=0)
                        inputs = torch.from_numpy(inputs).float().to('cuda')
                        y_pred = predict(model, inputs)
                        label = ctol[y_pred[0]]
                        
                        print(label)
                        
                        isDetected = True
                    isDetecting = False
                    
                # print("isDetecting: ", isDetecting)
                    
                cv2.imshow("Live From Webcam", processed_frame)
                processed_frames.append(processed_frame)
                
            else:
                print("Cannot Open Webcam, hw problem?")
                break

            # Press q and ESC to close window
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                cv2.destroyWindow("Live From Webcam")
                break
            
    except KeyboardInterrupt:
        print("Stream stopped")
    except Exception as e:
        print(e)
    inputStream.release()

    # data_df = pd.DataFrame(data=holistic_detector.landmarks[1:], columns=holistic_detector.landmarks[0])
    # holistic_detector.clear_logs()

    # # preprocessed data
    # data_df = data_df.drop(['index'], axis=1)
    # data_df = data_df[col_name]
    # processed_data = preprocess_data(data_df)
    # inputs = sample_data(processed_data)
    # inputs = np.expand_dims(inputs, axis=0)
    # inputs = torch.from_numpy(inputs).float().to('cuda')

    # y_pred = predict(model, inputs)
    # label = ctol[y_pred[0]]
    # print(label)
    # cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # outputStream = cv2.VideoWriter('{}_{}.mp4'.format(label, get_random_string(8)),
    #                         cv2_fourcc,
    #                         15, (INPUT_WIDTH, INPUT_HEIGHT))
    
    # for frame in processed_frames:
    #     # Write frame to outputStream
    #     outputStream.write(frame)
    # outputStream.release()