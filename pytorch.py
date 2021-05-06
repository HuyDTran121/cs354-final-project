import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import data_preprocess as dp
import pandas as pd

def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'model.th'))

class FaceDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,kernel_size=2)
        self.conv2 = nn.Conv2d(16,32,3)
        self.conv3 = nn.Conv2d(32,64,3)
        self.pool = nn.MaxPool2d(2,2)
        self.linear1 = nn.Linear(64 * 3 * 3, 64)
        self.linear2 = nn.Linear(64, 30)

    def forward(self, x):
        print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        print("here")
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x,1)
        x.F.relu(self.linear1(x))
        x.F.relu(self.linear2(x))

# Load in data

idlookup_df = pd.read_csv("./data/facial-keypoints-detection/IdLookupTable.csv")
train_df = pd.read_csv("./data/facial-keypoints-detection/training.csv")
test_df = pd.read_csv("./data/facial-keypoints-detection/test.csv")
# print(idlookup_df.head())
# print(train_df.head())
# print(test_df.head())

#drop empty data values and preprocess the csv to images and keypoints
unclean_train_df = train_df.fillna(method = 'ffill')
train_df.dropna(inplace=True)
train_images = dp.convert_data_to_image(train_df)
train_keypoints = dp.get_keypoints_features(train_df)

test_images = dp.convert_data_to_image(test_df)
test_keypoints =dp. get_keypoints_features(test_df)

unclean_train_images = dp.convert_data_to_image(unclean_train_df)
unclean_train_keypoints = dp.get_keypoints_features(unclean_train_df)

# print("Shape of train_images: {}".format(np.shape(train_images)))
# print("Shape of train_keypoints: {}".format(np.shape(train_keypoints)))
# print("Shape of test_images: {}".format(np.shape(test_images)))
# print("Shape of test_keypoints: {}".format(np.shape(test_keypoints)))

#just checking the data w/ plots
# sample_image_index = 20
# fig, axis = plt.subplots()
# plot_sample(train_images[sample_image_index], train_keypoints[sample_image_index], axis, "Sample image & keypoints")
# plt.show()

full_train_images = train_images
full_train_keypoints = train_keypoints
full_train_images = np.concatenate((full_train_images, unclean_train_images))
full_train_keypoints = np.concatenate((full_train_keypoints, unclean_train_keypoints))
print("Shape of train_images: {}".format(np.shape(full_train_images)))
print("Shape of train_keypoints: {}".format(np.shape(full_train_keypoints)))

#  Transform data into tensors
tensor_x = torch.Tensor(full_train_images)
tensor_y = torch.Tensor(full_train_keypoints)
tensor_dataset = TensorDataset(tensor_x, tensor_y)
dataloader = DataLoader(tensor_dataset)

# Set up model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = FaceDetector()
model.to(device)

loss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = .001, momentum=.9)

# Train

for epoch in range(50):
    print("epoch", epoch)
    model.train()
    for x,y in tensor_dataset:
        print(x.shape)
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        y_pred.to(device)
        loss_val = loss(y_pred, y)
        loss_val.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
    save_model(model)
    
