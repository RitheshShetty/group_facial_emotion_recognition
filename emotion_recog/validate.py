import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from emotion_recog.load_and_process import load_fer2013


images, emotion = load_fer2013(image_size=(64, 64))
images = images/255
print("Loading Emotion Recog Model")
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_mapping = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

emotion_pred = emotion_classifier.predict(images)
cm = confusion_matrix(np.argmax(emotion, axis=1), np.argmax(emotion_pred, axis=1))
cm_df = pd.DataFrame.from_records(cm)
cm_df.columns = emotion_mapping
cm_df.index = emotion_mapping
print("Confusion Matrix:\n", cm)

