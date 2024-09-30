import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks

from keras_preprocessing.image import ImageDataGenerator

trainDataGenerator = ImageDataGenerator(rescale=1./255, validation_split=0.2)
testDataGenerator = ImageDataGenerator(rescale=1./255)

train_d = trainDataGenerator.flow_from_directory(
     "./data/smiling_or_not",
      classes=["smile","non_smile"],
      color_mode="rgb",
      batch_size=16,
      target_size=(64,64),
      class_mode="categorical",
      subset="training")

valid_d = trainDataGenerator.flow_from_directory(
    "./data/smiling_or_not",
    classes=["smile","non_smile"],
    color_mode="rgb",
    batch_size=16,
    target_size=(64,64),
    class_mode="categorical",
    subset="validation"
)

test_d = testDataGenerator.flow_from_directory(
    "./data/smiling_or_not/test",
    color_mode="rgb",
    target_size=(64,64))

model = Sequential()
model.add(Conv2D(128,(3,3), input_shape=(64,64,3), padding="same", activation="gelu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3), padding="same", activation="gelu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64, activation="gelu"))
model.add(Dropout(0.2))
model.add(Dense(2, activation="softmax"))
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
callback_list=[
    callbacks.EarlyStopping(monitor="val_accuracy",patience=5,restore_best_weights=True),
    callbacks.ReduceLROnPlateau(factor=0.8,monitor="val_accuracy",patience=3)
]

history=model.fit(train_d,validation_data=valid_d,epochs=100,verbose=1,callbacks=callback_list)

model.save("../models/smile_model.h5")