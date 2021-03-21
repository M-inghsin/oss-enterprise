#!/usr/bin/python
#-*- coding: utf-8 -*-

# Library: pip3 install opencv-python
import cv2

# Load the cascade
# /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_alt.xml
face_cascade = cv2.CascadeClassifier('face_detector.xml')

# Read the input image
img = cv2.imread('img_test.jpg')

# Detect faces in the image
faces = face_cascade.detectMultiScale(img, 1.1, 4)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 250, 205), 2)

# Export the result
cv2.imwrite('img_test.png', img)
print('Found {0} face(s)!'.format(len(faces)), '\nSuccessfully saved')
