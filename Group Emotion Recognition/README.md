## Crowd-Emotion-Recognition

   <img src="emo.gif" width="640" height="400"></img>

**Overview**

Emotion Detection is a very improtant problem in the domain of advertisement as it can help the advertisement agency for choosing what to advertise during any occasion. Advertisement agencies spend millions of dollars to run a 30sec ad during the Super Bowl Sunday game. So it is very important to know the emotional state of the viewers when they are watching the game, so that the companies could display the ad as per the emotional state of the viewers at that very moment. Marketing companies and business owners spend a lot of money to analyze consumer feedback to products and services. Till date, surveys and reviews are the usual methods to gain the feedbacks but they are inaccurate and slow.

**Other Applications**
1) Movie Theatre: This model would work best for getting the reviews about a movie when setup in a theatre.
2) Classrooms: The emotions of the students recorded during the lecture could be a great feedback for a professor to understand how effective his/her teaching is and to restructure the course, lectures accordingly.
3) IT Companies: Employee feedback
4) A crucial application of this model could be in Health Care in understanding the facial expressions of a patient and to notify the doctors/physicians about the emergency situations.

**Model**

This project was aimed to develop a machine learning model for emotion detection of crowd using Deep Convolution Neural Network. The model contains 2 components:

1. FaceNet: This componnet is a pretrained model which uses Inception architecture for finding the faces in the images and generate bounding box coordiantes.
2. Emotion Classifier: It takes the cropped bounding box images as input and then perform emotion detection on the image to identify 7 emotions. Additionally, the model also predicts the crowd emotion as a whole.

**Setup**<br>
<br>Following libraries need to be installed<br>
* TensorFlow<br>
* Keras<br>
* Numpy<br>
* OpenCV<br>
* Pandas<br>

**Execution Steps:**<br>

* Run the facedetectV4.py file to test the model on a live stream video(real time video input) <br>
* The emotion recognition model is saved in a h5 file.<br>
* The model pipeline loads the emotion recognition model and outputs the emotions for each detected face in the frame along with Crowd Emotion at the top.<br>
* Press Q to quit
* After closing the application, a summary of the emotion detection is obtained in the form of a bar chart.
* Run /emotion_recog/validate.py file to get the confusion matrix.

**The model gives NA as an ouput when there is no person in the frame.** 









