import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_preprocess as dp
import keras

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

#Model Arch
model = Sequential()

model.add(BatchNormalization(input_shape=(96, 96, 1)))
model.add(Convolution2D(24, 5, 5, border_mode="same", init= "he_normal", input_shape=(96, 96, 1), dim_ordering="tf"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))

model.add(Convolution2D(36, 5, 5))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))

model.add(Convolution2D(48, 5, 5))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))

model.add(Convolution2D(64, 3, 3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))

model.add(Convolution2D(64, 3, 3))
model.add(Activation("relu"))

model.add(GlobalAveragePooling2D())

model.add(Dense(500, activation="relu"))
model.add(Dense(90, activation="relu"))
model.add(Dense(30)) 

model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='face_model.h5', verbose=1, save_best_only=True)
epochs = 30
hist = model.fit(full_train_images, full_train_keypoints, validation_split=0.2, shuffle=True, epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)