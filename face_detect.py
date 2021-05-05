import cv2
import image_processor

import os.path
from os import path

image = image_processor.loadImage("./data/face_1.jpg")
# print(path.exists("./data/face_1.jpg"))

# cv2.imshow('face_1', image)


grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
bounding_boxes = face_cascade.detectMultiScale(grayscale_image, scaleFactor = 1.2, minNeighbors = 5)
print(type(face_cascade))
print(type(bounding_boxes))
print(bounding_boxes)

for (x, y, w, h)  in bounding_boxes:
  cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('face_1', image)

#waits for user to press any key 
#(this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0) 
  
#closing all open windows 
cv2.destroyAllWindows()