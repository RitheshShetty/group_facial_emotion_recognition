# courtsey: Andrew ng  Convolutional Neural networks couse

import warnings

warnings.filterwarnings('ignore')
from keras import backend as K
import datetime

K.set_image_data_format('channels_first')
from keras.models import load_model
from fr_utils import *
from inception_blocks_v2 import *
import matplotlib.pyplot as plt
calibrate = 1

def facedet(frame, net):
    frame = cv2.resize(frame, (400, 400))
    (h, w) = frame.shape[:2]
    faceblob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
    net.setInput(faceblob)
    detections = net.forward()
    return detections


def days_between(d1, d2):
    d1 = datetime.datetime.strptime(d1, '%m-%d-%Y')
    d2 = datetime.datetime.strptime(d2, '%m-%d-%Y')
    return abs((d2 - d1).days)


def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_sum(basic_loss)

    return loss


print("Loading Emotion Recog Model")
emotion_model_path = 'emotion_recog/models/_mini_XCEPTION.102-0.66.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_mapping = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

emotion_count = {"Angry": 0, "Disgust":  0, "Fear": 0, "Happy": 0, "Sad": 0, "Surprise": 0, "Neutral": 0}

Maxcount = 50
Rqconfidence = 0.6
print("[INFO] loading face detection model...")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
print("Loading Inception net Model")
FRmodel = faceRecoModel(input_shape=(3, 96, 96))
print("Compiling Model ....")
FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
print("Loading weights ....")
FRmodel.load_weights("model_weights/FRmodel.h5")
# load_weights_from_FaceNet(FRmodel)
print("Loaded weights ....")
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

cv2.waitKey(0)
while True:
    ret, frame = cap.read()
    detections = facedet(frame, net)
    frame2 = cv2.resize(frame, (400, 400))
    (h, w) = frame2.shape[:2]
    for i in range(0, detections.shape[2]):  # iterate  over the detected faces
        confidence = detections[0, 0, i, 2]
        if confidence < Rqconfidence:  # pass the face if the detection in the blob is more than the confidence
            continue
        box1 = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        scale = [-30, -80, 30, 30]  # for selecting only the part of detected face
        boxfloat = box1 + scale
        box = boxfloat.astype(int)
        (startX, startY, endX, endY) = box.astype("int")
        im2 = frame2[box[1]:box[3], box[0]:box[2]]  # grab the detected face
        if im2.size != 0:
            resized = cv2.resize(im2, (96, 96))
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face_gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)[startY:endY, startX:endX]
            face_gray = cv2.resize(face_gray, (64, 64)) / 255
            face_gray = np.expand_dims(np.expand_dims(face_gray, axis=2), axis=0)
            emotion_pred = emotion_classifier.predict(face_gray)
            emotion_score = np.max(emotion_pred)
            emotion = emotion_mapping[np.argmax(emotion_pred)]
            emotion_count[emotion] += 1
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame2, (startX, startY), (endX, endY), (0, 0, 255), 2)
            text_emotion = emotion + "-" + str(np.round(emotion_score, 2))
            cv2.putText(frame2, text_emotion, (startX, endY + 20), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
cap.release()
cv2.destroyAllWindows()

plt.bar(list(emotion_count.keys()), emotion_count.values(), align='center', alpha=0.5)
plt.xticks(list(emotion_count.keys()), list(emotion_count.keys()))
plt.ylabel('Emotion Count') 
plt.title('Emotion')
plt.show()

