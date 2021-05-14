
from torch import load
from os import path
from pytorch import FaceDetector
import image_processor
import cv2
import torch
import numpy as np
model = FaceDetector()
model.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'model.th')))
print("Loaded")
image = image_processor.loadImage("./data/test_face.png")
grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
tensor = torch.from_numpy(np.reshape(grayscale_image, (1,1,96,96))).float()
prediction = model(tensor).detach().numpy()

print(prediction)
print(np.shape(image))
for i in range(2):
    x = int(prediction[0][i*2])
    y = int(prediction[0][i*2+1])
    print(x,y)
    image[y][x] = [255,0,255,1]
cv2.imshow("image", image)
cv2.waitKey(0)