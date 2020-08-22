import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("model/haarcascade_frontalface_alt2.xml");

faceSamples = []
userIds = []

# function to get the images and label data
def trainImages(path, user_id):
    for img_path in os.listdir(path):
        if '.DS_Store' in img_path:
            continue
        pil_img = Image.open(img_path).convert('L') # convert it to grayscale
        img_numpy = np.array(pil_img, 'uint8')
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            userIds.append(user_id)

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")

trainImages('dataset/1', 1)
trainImages('dataset/2', 2)

recognizer.train(faceSamples, np.array(userIds))

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml')  # for mac
# recognizer.save()  # for linux

# Print the numer of faces trained and end program
print("\n [INFO] faces trained. Exiting Program")
