from unittest import result
import cv2
from collections import Counter
from deepface import DeepFace
from openFACS import sendAUS

expressions: dict = {
    "neutral": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "angry":   [0.0, 0.0, 5.0, 3.0, 0.0, 4.5, 4.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "disgust": [0.0, 0.0, 5.0, 0.0, 1.3, 2.0, 2.5, 5.0, 0.0, 0.0, 4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "fear":    [5.0, 2.7, 5.0, 4.0, 0.0, 4.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.5, 0.0, 4.0, 1.0, 0.0, 0.0],
    "surprise":[4.0, 4.2, 0.0, 3.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.9, 0.0, 0.0],
    "sad":     [5.0, 0.0, 5.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 4.5, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "happy":   [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.8, 0.0, 0.0]
}

cap = cv2.VideoCapture(-1)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

emotion_list = []

while True:
    ret,frame = cap.read() #reads an image from video
    try:
        result = DeepFace.analyze(frame, actions=['emotion'])
    except:
        continue
    
    emotion = result['dominant_emotion']

    # shows the camera image
    #cv2.imshow('Original video', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

    emotion_list.append(emotion)
    
    # collect 5 emotions to smooth the reaction
    if (len(emotion_list)==5):
        counts = Counter(emotion_list)
        target = max(counts, key=counts.get)
        print(counts, target)

        #send the emotion smothed to the avatar interface
        AU = expressions[target]
        sendAUS(AU, 0.5)

        #empty list to catch and smooth the next emotion
        emotion_list = []

cap.release()
cv2.destroyAllWindows()