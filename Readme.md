# Dynamic Hand Gesture Recognition on Thai Sign Language

## Background
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;หนึ่งในวิธีที่ผู้พิการทางการได้ยินใช้ในการสื่อสารกับผู้อื่นคือภาษามือ แต่มักประสบปัญหาในการสือสารกับบุคคลทั่วไปที่ไม่มีความเข้าใจในภาษามือ ในโครงการนี้จึงได้มีการนำเทคโนโลยีทางด้านคอมพิวเตอร์ที่เรียกว่า คอมพิวเตอร์วิทัศน์ หรือ Computer Vision เข้ามาช่วยในการวิเคราะห์ความหมายของภาษามือแต่ละท่าผ่านการดูภาพที่แสดงท่าทางภาษามือเพื่อช่วยแก้ปัญหาการสื่อสารกับผู้พิการทางการได้ยินให้มีความสะดวกมากยิ่งขึ้น

## Scope
- คำในภาษาไทยจำนวน 24 คำ ได้แก่ `1.ฉัน, 2.เธอ, 3.เขา, 4.พวกเรา, 5.สวัสดี, 6.ชื่อ, 7.ผู้ชาย, 8.ผู้หญิง, 9.ความรัก, 10.ชอบ, 11.ไม่ชอบ, 12.ดีใจ, 13.เสียใจ, 14.โกรธ, 15.ร้องไห้, 16.ยาก, 17.ง่าย, 18.คิดถึง, 19.รองเท้า, 20.กางเกงยีนส์, 21.หมวก, 22.แว่นตา, 23.นาฬิกาข้อมือ, และ 24.ผ้าเช็ดหน้า`
- การแสดงผลลัพธ์แบบ real-time


## Technical Challenges
* การเลือกใช้วิธีการประมวลผลให้รองรับการทำงานแบบ real-time และ มีความถูกต้องในการทำงาน
* ปริมาณข้อมูลภาษามือที่เป็นภาษาไทยมีจำนวนน้อย

## Related Works
- Recent Advances in Video-Based Human Action Recognition using Deep Learning: A Review
(IEEE Xplore, 03 July 2017)
>> ศึกษาการประยุกต์ใช้ Deep Learning ในการวิเคราะห์ท่าทางการเคลื่อนไหวจากภาพใน 3 มุมมอง ได้แก่ 1.Single Viewpoint 2.Multiple Viewpoint 3.RGB-Depth (ใช้อุปกรณ์พิเศษ เช่น Microsoft Kinect)
- SubUNets: End-To-End Hand Shape and Continuous Sign Language Recognition (IEEE Xplore, 25 December 2017)
>> ศึกษาการประยุกต์ใช้ Convolutional Neural Network และ LSTM Deep Learning Model ในการวิเคราะห์ลำดับของภาษามือที่มีความต่อเนื่องกัน
- Dynamic Hand Gesture Recognition Using Computer Vision and Neural Networks (IEEE Xplore, 11 November 2018)
>> ประยุกต์ใช้ Image Processing ในการ pre-process ภาพ เช่น Gaussian Mixture-based Background/Foreground Segmentation, Image Thresholding เพื่อสร้าง Motion History Image และนำไปประมวลผลด้วย Neural Network Model
- Video-based isolated hand sign language recognition using a deep cascaded model (Springer
Link, 02 June 2020)
>> วิเคราะห์ภาษามือด้วย Deep Cascaded Model 3 ส่วน ได้แก่ 1. Single Shot Detector (SSD) ในการทำ Hand Detection 2.Convolutional Neural Network (CNN) เพื่อทำ Feature Extraction และ 3.Long Shot Term Memory (LSTM) เพื่อเรียนรู้และประมวลผลความหมายของท่าทางต่างๆ
- Hand Gesture Recognition for Thai Sign Language in Complex Background Using Fusion of
Depth and Color Video (ScienceDirect, 24 May 2016)
>> วิเคราะห์ภาษามือแบบ Fingerspelling ด้วยการประยุกต์ใช้ Image Processing เช่น Image Segmentation, HOG Feature Extraction ร่วมกับอุปกรณ์พิเศษ Microsoft Kinect และประมวลผลด้วย Neural Network Model
- Thai Sign Language Recognition Using 3D Convolutional Neural Networks (ACL Digital Library, 27 July 2019)
>> วิเคราะห์ภาษามือจำนวน 64 ท่าทางโดยการวิเคราะห์ข้อมูลจากอุปกรณ์ Microsoft Kinect ได้แก่ Depth, Color, Skeleton, Hand Shapes, Body Movement และประมวลผลด้วย 3D Convolutional Neural Network Model
- Thai Sign Language Recognition: an Application of Deep Neural Network
(IEEE Xplore, 11 May 2021)
>> วิเคราะห์ภาษามือจำนวน 5 ท่าทางโดยการทำ Hand Landmark Extraction ด้วย Mediapipe Model และทดสอบการประมวลผลด้วย Recurrent Neural Network (RNN) Model จำนวน 3 ตัว ได้แก่ LSTM, BiLSTM, และ GRU

## Method and Results

## Discussion and Future Work