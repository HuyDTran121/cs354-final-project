
from torch import load
from os import path
from pytorch import FaceDetector
import image_processor
import cv2
r = FaceDetector()
r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'model.th'), map_location='cpu'))
print("Loaded")
image = image_processor.loadImage("./data/test_face.png")
grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
r.predict()
prediction = r(grayscale_image)
