import cv2
import os
from image_edit import resize_image

face_detector = cv2.CascadeClassifier('model/haarcascade_frontalface_alt2.xml')

def cutFaceImages(src_dir, dst_dir):
    # create the dest save dir
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for img_path in os.listdir(src_dir):
        img = cv2.imread(src_dir + '/' + img_path)
        # is image ?
        if not hasattr(img, 'shape'):
            print(img_path, 'is not a image file!')
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(30, 30)
            )
        index = 0
        for (x,y,w,h) in faces:
            index += 1
            gray = gray[y:y+h, x:x+w]
            if (0,0) == gray.shape:
                continue
            gray = resize_image(gray, 64)
            save_path = dst_dir + '/' + str(index) + '-' + img_path
            cv2.imwrite(save_path, gray)


cutFaceImages('collect/1', 'dataset/1')


