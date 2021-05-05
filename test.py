#  TESTING CODE FROM https://google.github.io/mediapipe/solutions/face_mesh#python-solution-api

import cv2
import mediapipe as mp
import numpy as np
import image_processor
import face_detect
import os.path
from os import path
from mediapipe.framework.formats import landmark_pb2
import math
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

glasses = image_processor.loadImage("./data/glasses.png")
print(path.exists("./data/face_1.jpg"))

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # image[0][0] = [255,255,255]
    if results.multi_face_landmarks:
      height, width,_ = np.shape(image)
      lx = int(results.multi_face_landmarks[0].landmark[159].x * width)
      ly = int(results.multi_face_landmarks[0].landmark[159].y * height)
      rx = int(results.multi_face_landmarks[0].landmark[386].x * width)
      ry = int(results.multi_face_landmarks[0].landmark[386].y * height)
      angle = -math.atan((ry-ly)/(rx-lx)) * 360 / (2*math.pi)
      print(angle)
      rot_mat = cv2.getRotationMatrix2D(tuple(np.array(glasses.shape[1::-1]) / 2), angle, 1)
      rotated = cv2.warpAffine(glasses, rot_mat, glasses.shape[1::-1], flags=cv2.INTER_LINEAR)

      bounding_boxes = face_detect.detect_face(image)
      for (x, y, w, h)  in bounding_boxes:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
      image = image_processor.drawImage(rotated, image, lx-100, ly-50)
    #   for face_landmarks in results.multi_face_landmarks:
    #     mp_drawing.draw_landmarks(
    #         image=image,
    #         landmark_list=face_landmarks,
    #         connections=mp_face_mesh.FACE_CONNECTIONS,
    #         landmark_drawing_spec=drawing_spec,
    #         connection_drawing_spec=drawing_spec)
    cv2.imshow('MediaPipe FaceMesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()