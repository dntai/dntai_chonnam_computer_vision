# COMPUTER VISION FINAL PROJECT
Chonnam National University, Gwangju, South Korea<br/>
**[2018-06] Computer Vision Class**<br/>
Professor: **Chilwoo Lee**<br/>
Student: **Do Nhu Tai**<br/>
**Target: Emotional Expression with Bag-of-Word and Deep Learning**<br/>

## Dataset: [Kaggle Fer2013] - Facial Expression Recognition Challenge
<a href="https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/">https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/</a></br>
(1) Dataset Information:<br/>
<pre>
Number of images in the training dataset:	 28,709
Number of images in the public dataset:		  3,589
Number of images in the private dataset:	  3,589
Image information: 48 x 48 x 1
</pre>
(2) Training Images:<br/>
![alt text](https://github.com/dntai/dntai_chonnam_computer_vision/blob/master/images/training_images.png)

(3) Validating Images:<br/>
![alt text](https://github.com/dntai/dntai_chonnam_computer_vision/blob/master/images/validating_images.png)

(4) Testing Images:<br/>
![alt text](https://github.com/dntai/dntai_chonnam_computer_vision/blob/master/images/testing_images.png)

(5) Data distribution:<br/>
![alt text](https://github.com/dntai/dntai_chonnam_computer_vision/blob/master/images/histogram_images.png)

## Problem 01: Emotion with Bag-of-Word and Sparse SIFT Feature

(1) Traininig Process:<br/>
+ Extract descriptors using SIFT<br/>
+ Merge descriptors into local patches<br/>
+ Cluster local patches using Mini Batch K-Means to build codewords<br/>
+ Build Feature Histogram Model based on codewords<br/>
+ Normalize Feature Histogram Model
+ Classify by Multi-Class SVM<br/>

(2) Testing Process:<br/>
+ Extract descriptors using SIFT<br/>
+ Match descriptors with K-Means Cluster Center to build codewords for testing image<br/>
+ Create and normalize Feature Histogram Model for Image Codewords<br/>
+ Predict by Multi-Class SVM<br/>

![alt text](https://github.com/dntai/dntai_chonnam_computer_vision/blob/master/images/BagOfWord_Model.png)

(3) Results:<br/>
+ Number of Codeword Cluster = 4000<br/>


## Problem 02: Emotion with Deep Learning

## Problem 03: Video Emotion Extraction

## References

## Personal information
**Do Nhu Tai**<br/>
Supervisor: Professor **Kim Soo-Hyung**<br/>
Pattern Recognition Lab<br/>
**Chonnam National University, Korea**<br/>
E-mail: donhutai@gmail.com<br/>