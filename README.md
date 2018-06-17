# COMPUTER VISION FINAL PROJECT
Chonnam National University, Gwangju, South Korea<br/>
**[2018-06] Computer Vision Class**<br/>
Professor: **Chilwoo Lee**<br/>
Student: **Do Nhu Tai**<br/>
**Target: Emotional Expression with Bag-of-Word and Deep Learning**<br/>

## Dataset: [Kaggle Fer2013] - Facial Expression Recognition Challenge
<a href="https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/">https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/</a></br>
**(1) Dataset Information:**<br/>
<pre>
Number of images in the training dataset:	 28,709
Number of images in the public dataset:		  3,589
Number of images in the private dataset:	  3,589
Image information: 48 x 48 x 1
</pre>
**(2) Training Images:**<br/>
![alt text](https://github.com/dntai/dntai_chonnam_computer_vision/blob/master/images/training_images.png)

**(3) Validating Images:**<br/>
![alt text](https://github.com/dntai/dntai_chonnam_computer_vision/blob/master/images/validating_images.png)

**(4) Testing Images:**<br/>
![alt text](https://github.com/dntai/dntai_chonnam_computer_vision/blob/master/images/testing_images.png)

**(5) Data distribution:**<br/>
![alt text](https://github.com/dntai/dntai_chonnam_computer_vision/blob/master/images/histogram_images.png)

## Problem 01: Emotion with Bag-of-Word and Sparse SIFT Feature

**(1) Traininig Process:**<br/>
+ Extract descriptors using SIFT<br/>
+ Merge descriptors into local patches<br/>
+ Cluster local patches using Mini Batch K-Means to build codewords<br/>
+ Build Feature Histogram Model based on codewords<br/>
+ Normalize Feature Histogram Model
+ Classify by Multi-Class SVM<br/>

**(2) Testing Process:**<br/>
+ Extract descriptors using SIFT<br/>
+ Match descriptors with K-Means Cluster Center to build codewords for testing image<br/>
+ Create and normalize Feature Histogram Model for Image Codewords<br/>
+ Predict by Multi-Class SVM<br/>

![alt text](https://github.com/dntai/dntai_chonnam_computer_vision/blob/master/images/BagOfWord_Model.png)

**(3) Results:**<br/>
+ Number of Codeword Cluster = 4000<br/>
![alt text](https://github.com/dntai/dntai_chonnam_computer_vision/blob/master/images/BagOfWord_Result.png)

## Problem 02: Emotion with Deep Learning

**(1) Deep Learning CNN-Like-VGG16 Model:**<br/>
<pre>
+ Block 1 – 3 Conv2D (64, (3,3)) , MaxPooling (2,2), Dropout (0.2)
+ Block 2 – 4 Conv2D (128, (3,3)), MaxPooling (2,2), Dropout (0.2)
+ Block 3 – 4 Conv2D (256, (3,3)), MaxPooling (2,2), Dropout (0.2)
+ Classifier – Flattern, Dense (1024), Dropout(0.5), Dense(7, SoftMax)
</pre>

**(2) Training History:**<br/>
![alt text](https://github.com/dntai/dntai_chonnam_computer_vision/blob/master/images/CNN_Training.png)

**(3) Results:**<br/>
![alt text](https://github.com/dntai/dntai_chonnam_computer_vision/blob/master/images/CNN_Result.png)

## Problem 03: Video Emotion Extraction

**(1) Program Description:**<br/>
+ Extract small video clips in a video and vote video emotion of small clips<br/>
![alt text](https://github.com/dntai/dntai_chonnam_computer_vision/blob/master/images/VideoEmotionExtraction.png)

**(2) Program Feature:**<br/>
+ Console Program with Modules <br/>
+ Emotion MLCNN, <br/>
+ Face Detection Dlib, OpenCV, MTCNN, <br/>
+ Face Matching using Hungarian Method, <br/>
+ Face Description with VGG Face<br/>


## References

## Personal information
**Do Nhu Tai**<br/>
Supervisor: Professor **Kim Soo-Hyung**<br/>
Pattern Recognition Lab<br/>
**Chonnam National University, Korea**<br/>
E-mail: donhutai@gmail.com<br/>