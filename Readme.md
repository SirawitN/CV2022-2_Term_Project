# Dynamic Hand Gesture Recognition on Thai Sign Language

## Background

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;หนึ่งในวิธีที่ผู้พิการทางการได้ยินใช้ในการสื่อสารกับผู้อื่นคือภาษามือ แต่มักประสบปัญหาในการสือสารกับบุคคลทั่วไปที่ไม่มีความเข้าใจในภาษามือ ในโครงการนี้จึงได้มีการนำเทคโนโลยีทางด้านคอมพิวเตอร์ที่เรียกว่า คอมพิวเตอร์วิทัศน์ หรือ Computer Vision เข้ามาช่วยในการวิเคราะห์ความหมายของภาษามือแต่ละท่าผ่านการดูภาพที่แสดงท่าทางภาษามือเพื่อช่วยแก้ปัญหาการสื่อสารกับผู้พิการทางการได้ยินให้มีความสะดวกมากยิ่งขึ้น

## Scope

- คำในภาษาไทยจำนวน 24 คำ ได้แก่ `1.ฉัน, 2.เธอ, 3.เขา, 4.พวกเรา, 5.สวัสดี, 6.ชื่อ, 7.ผู้ชาย, 8.ผู้หญิง, 9.ความรัก, 10.ชอบ, 11.ไม่ชอบ, 12.ดีใจ, 13.เสียใจ, 14.โกรธ, 15.ร้องไห้, 16.ยาก, 17.ง่าย, 18.คิดถึง, 19.รองเท้า, 20.กางเกงยีนส์, 21.หมวก, 22.แว่นตา, 23.นาฬิกาข้อมือ, และ 24.ผ้าเช็ดหน้า`
- การแสดงผลลัพธ์แบบ real-time

<p align="center">
<img src="/resources/readme/hand_sign_language.jpg">
</p>

## Technical Challenges

- การเลือกใช้วิธีการประมวลผลให้รองรับการทำงานแบบ real-time และ มีความถูกต้องในการทำงาน
- ตำแหน่งของผู้ทำภาษามือในกล้องที่ต่างกัน
- ปริมาณข้อมูลภาษามือที่เป็นภาษาไทยมีจำนวนน้อย

## Related Works

- Recent Advances in Video-Based Human Action Recognition using Deep Learning: A Review
  (IEEE Xplore, 03 July 2017)
  > > ศึกษาการประยุกต์ใช้ Deep Learning ในการวิเคราะห์ท่าทางการเคลื่อนไหวจากภาพใน 3 มุมมอง ได้แก่ 1.Single Viewpoint 2.Multiple Viewpoint 3.RGB-Depth (ใช้อุปกรณ์พิเศษ เช่น Microsoft Kinect)
- SubUNets: End-To-End Hand Shape and Continuous Sign Language Recognition (IEEE Xplore, 25 December 2017)
  > > ศึกษาการประยุกต์ใช้ Convolutional Neural Network และ LSTM Deep Learning Model ในการวิเคราะห์ลำดับของภาษามือที่มีความต่อเนื่องกัน
- Dynamic Hand Gesture Recognition Using Computer Vision and Neural Networks (IEEE Xplore, 11 November 2018)
  > > ประยุกต์ใช้ Image Processing ในการ pre-process ภาพ เช่น Gaussian Mixture-based Background/Foreground Segmentation, Image Thresholding เพื่อสร้าง Motion History Image และนำไปประมวลผลด้วย Neural Network Model
- Video-based isolated hand sign language recognition using a deep cascaded model (Springer
  Link, 02 June 2020)
  > > วิเคราะห์ภาษามือด้วย Deep Cascaded Model 3 ส่วน ได้แก่ 1. Single Shot Detector (SSD) ในการทำ Hand Detection 2.Convolutional Neural Network (CNN) เพื่อทำ Feature Extraction และ 3.Long Shot Term Memory (LSTM) เพื่อเรียนรู้และประมวลผลความหมายของท่าทางต่างๆ
- Hand Gesture Recognition for Thai Sign Language in Complex Background Using Fusion of
  Depth and Color Video (ScienceDirect, 24 May 2016)
  > > วิเคราะห์ภาษามือแบบ Fingerspelling ด้วยการประยุกต์ใช้ Image Processing เช่น Image Segmentation, HOG Feature Extraction ร่วมกับอุปกรณ์พิเศษ Microsoft Kinect และประมวลผลด้วย Neural Network Model
- Thai Sign Language Recognition Using 3D Convolutional Neural Networks (ACL Digital Library, 27 July 2019)
  > > วิเคราะห์ภาษามือจำนวน 64 ท่าทางโดยการวิเคราะห์ข้อมูลจากอุปกรณ์ Microsoft Kinect ได้แก่ Depth, Color, Skeleton, Hand Shapes, Body Movement และประมวลผลด้วย 3D Convolutional Neural Network Model
- Thai Sign Language Recognition: an Application of Deep Neural Network
  (IEEE Xplore, 11 May 2021)
  > > วิเคราะห์ภาษามือจำนวน 5 ท่าทางโดยการทำ Hand Landmark Extraction ด้วย Mediapipe Model และทดสอบการประมวลผลด้วย Recurrent Neural Network (RNN) Model จำนวน 3 ตัว ได้แก่ LSTM, BiLSTM, และ GRU

## Method and Results

### Data Preparation

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;เริ่มต้นจากการศึกษาท่าทางภาษามือและทำการคัดเลือกท่าที่จะนำมาใช้ในโครงการ โดยเลือกท่าทางที่มีลักษณะแตกต่างกันเพื่อให้มีความหลากหลาย ทั้งท่าทางที่ใช้ 1 และ 2 มือ ท่าที่มีการขยับแขนขึ้น-ลง ท่าที่มีการใช้ศีรษะประกอบการทำ
[สอนภาษามือ - TK Channel](https://www.youtube.com/playlist?list=PL04-r7CQK5w9BPtNWXnAccIm0zdO31PDy)

<p align="center">
<img src="/resources/readme/hand_sign_language.jpg">
</p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;แต่เนื่องจากข้อมูลที่สามารถนำมาใช้ในกระบวนการ train โปรแกรม Machine Learning Model นั้นมีจำนวนน้อย จึงได้มีการเก็บภาพของแต่ละท่าทางเพิ่ม ท่าละ 10 ภาพ
<p align="center">
<img src="/resources/readme/hand_sign.jpg">
</p>

### Application Workflow

<p align="center">
<img src="/resources/readme/workflow_diagram.png">
</p>

#### Data Split  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ทำการแบ่ง dataset ออกเป็น train dataset 80% และ test dataset 20% โดยให้อัตราส่วนของท่าภาษามือใน train และ test set เท่ากัน

#### OpenCV

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;รับข้อมูลจากกล้อง Webcam ผ่านการตั้งค่าด้วยไลบราลี่ OpenCV

#### Mediapipe

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;เป็น open-source Machine Learning Platform เพื่อให้บริการเครื่องมือที่เกี่ยวข้องกับ Machine Learning เพื่อตอบสนองต่อการใช้งานด้านการ live หรือ streaming โดยในโครงการนี้ได้มีการเลือกใช้เครื่องมือที่มีชื่อว่า Mediapipe Holistic pipeline ทีมีการทำงานร่วมกันของ Mediapipe Pose, Face, และ Hand เพื่อทำหน้าที่ในการตรวจจับจุด landmark บริเวณตัว ใบหน้า และมือ ซึ่งในโครงการนี้จะเลือกใช้ข้อมูลจุด landmark ของบริเวณลำตัวและมือในการประมวลผล โดยจุด landmark บริเวณลำตัวและมือจะมีจำนวน 33 และ 21 จุดตามลำดับ

<p align="center">
<img src="/resources/readme/pose_landmarks.png">
<img src="/resources/readme/hand_landmarks.png">
</p>

#### Data Preprocess

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ภายหลังการประมวลผลจุด landmark ด้วย Mediapipe จะได้ผลลัพธ์เป็นตำแหน่งของ landmark แต่ละจุด โดยถูก normalized ให้มีค่าอยู่ในช่วง `[0.0, 1.0] `ด้วยความกว้างและความสูงของรูปภาพ  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;หลังจาก normalize ตำแหน่ง keypoint ของแต่ละ frame แล้ว ทุกตำแหน่งจะถูกลบด้วยตำแหน่งของจมูกจาก Mediapipe Pose (ตำแหน่งที่ 0) เพื่อให้จุด `(0,0)` ของ keypoint อยู่ที่จมูกแทน เพื่อให้ invariant ต่อตำแหน่งของผู้ทำภาษามือใน frame  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;จากนั้น เราจะนำความกว้างของไหล่ (ระยะห่างระหว่าง keypoint ตำแหน่งที่ 11 และ 12 จาก Mediapipe Pose) มาหารจาก keypoint ทุกตำแหน่ง เพื่อเป็นการ normalize ตำแหน่งของ และเพื่อให้ invariant ต่อขนาดของผู้ทำภาษามือใน frame

#### Keyframe

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ข้อมูลที่ได้จากการประมวลผลด้วย Mediapipe นั้นจะประกอบไปด้วยตำแหน่งของจุด landmark จำนวน 75 จุด โดยเป็นจุด landmark ของมือจำนวน 42 จุด โดยหากมีการตรวจจับตำแหน่งของมือได้ตั้งแต่ 1 ข้างขึ้นไปเป็นจำนวนมากกว่า 10 เฟรมติดต่อกัน จึงจะนับว่าเป็น keyframe และเริ่มการประมวลผล โดยจะหยุดการประมวลผลเมื่อตรวจจับตำแหน่งมือไม่ได้เป็นจำนวน 10 เฟรมขึ้นไป
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ช่วงของเฟรมทั้งหมดที่ถูกประมวลผล จะถูกกรองเอา frame ที่ไม่ติด keypoint ของมือเลยออก จากนั้น frame ที่ถูกกรองแล้วจะถูกนำไป sampling มาเพียง 30 frame โดยใช้ระยะห่างเท่า ๆ กัน (ใช้ `np.linspace`)

#### LSTM Classification Model

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model ที่เลือกใช้คือ Recurrent Neural Network (RNN) ที่ชื่อ Long Short-Term Memory (LSTM) มี architecture ดังรูป  
<p align="center">  
<img src="/resources/readme/model_architecture.jpg">  
</p>  
โดยใช้ parameters ดังนี้  

```
{
  'optimizer': 'AdamW' with learning rate = 1e-4
  'scheduler': 'ReduceLROnPlateau'
  'critirion': 'CrossEntropyLoss'
}
```

#### UI

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;โปรแกรมใช้ภาษา Python ในการทำงานและใช้ package ที่ชื่อ [Gradio](https://gradio.app/) ในการทำ UI Components

### Demonstration

## Discussion and Future Work
