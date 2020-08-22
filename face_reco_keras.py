import cv2
import numpy as np
import os
import keras
from matplotlib import pyplot as plt
from image_edit import resize_image

# recognize the face in pictures
cascadePath = "model/haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(cascadePath);

img = cv2.imread ("/Users/magina/Downloads/anya.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.2, 
            minNeighbors=5, 
            minSize=(30, 30)
        )

print('how many faces in this picture? ', len(faces))

# use the keras model to predict
model = keras.models.load_model("face.h5")

X = []
position = []
for(x,y,w,h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # pick up the face and validate the size
    gray = gray[y:y+h, x:x+w]
    if (0, 0) == gray.shape:
        continue 

    # make sure the size is 64*64
    gray = resize_image(gray, 64, True)

    X.append(gray)
    position.append((x, y))

# the final face num
n = len(X)
X = np.array(X)
X = X.reshape(n, 64, 64, 1)

predicts = model.predict(X)

names = ['Unknown', 'Magina', 'Anya'] 
for i in range(n):
    plt.grid(False)
    plt.imshow(X[i], cmap="gray")
    plt.title('face index: ' + str(i))
    user_id = np.argmax(predicts[i])
    plt.xlabel('predicted user: ' + names[user_id])
    plt.show()

    # 写入预测人名
    x, y = position[i]
    cv2.putText(img, names[user_id], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


cv2.imshow('result', img) 
cv2.waitKey(0)

cv2.destroyAllWindows()
