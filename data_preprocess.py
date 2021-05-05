import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#credit to mohamedahmedae on kaggle
img_height = 96
img_width = 96
def convert_data_to_image(image_data):
    images = []
    for _, sample in image_data.iterrows():
        image = np.array(sample["Image"].split(' '), dtype=int)
        image = np.reshape(image, (img_height,img_width,1))
        images.append(image)
    images = np.array(images)/255
    return images
def get_keypoints_features(keypoint_data):
    keypoint_data = keypoint_data.drop("Image", axis=1)
    keypoint_features = []
    for _, sample_keypoints in keypoint_data.iterrows():
        keypoint_features.append(sample_keypoints)
    
    keypoint_features = np.array(keypoint_features, dtype="float")
    return keypoint_features
def plot_sample(image, keypoint, axis, title):
    image = image.reshape(img_height, img_width)
    axis.imshow(image, cmap="gray")
    axis.scatter(keypoint[::2], keypoint[1::2], marker='x', s=20)
    plt.title(title)





