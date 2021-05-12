from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
# import sys

InitialLR = 1e-4
NumberOfEpochs = 20
BatchSize = 32

print("<---- Loading images ---->")
imagePaths = list(paths.list_images(
    r"C:\code\Python\OpenCV\Mask Detection\dataset"))
data = []
labels = []

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]

    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    data.append(image)
    labels.append(label)

classes = set(labels)
data = np.array(data, dtype="float32")
labels = np.array(labels)

labels = labels.reshape(-1, 1)

ohe = OneHotEncoder()
labels = ohe.fit_transform(labels).toarray()

(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.10, stratify=labels, random_state=42)

aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

print("<---- Compiling model ---->")
opt = Adam(lr=InitialLR, decay=InitialLR / NumberOfEpochs)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

print("<---- Training head ---->")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BatchSize),
    steps_per_epoch=len(trainX) // BatchSize,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BatchSize,
    epochs=NumberOfEpochs)

print("<---- Evaluating network ---->")
predIdxs = model.predict(testX, batch_size=BatchSize)
predIdxs = np.argmax(predIdxs, axis=1)

print("<---- Saving mask detector model ---->")
model.save('final_mask_detector.model', save_format="h5")

#----------------------Classification Report---------------------------

# sys.stdout = open("./report/Report(70-30).txt", "w")
# print(classification_report(testY.argmax(axis=1), predIdxs, target_names=classes))
# sys.stdout.close()

#------------------------------Plotting--------------------------------

# N = 20
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
# plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
# plt.title("Training Loss and Accuracy (70-30)")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")
# plt.savefig('plot70-30.png')
