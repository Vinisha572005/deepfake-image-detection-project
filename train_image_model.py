import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

# ---------------------------------------------------
# IMAGE SETTINGS
# ---------------------------------------------------

IMG_SIZE = 128
BATCH_SIZE = 32

# ---------------------------------------------------
# DATA GENERATOR
# ---------------------------------------------------

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# ---------------------------------------------------
# TRAIN DATA
# ---------------------------------------------------

train_data = datagen.flow_from_directory(
    'datasets/images',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

# ---------------------------------------------------
# VALIDATION DATA
# ---------------------------------------------------

val_data = datagen.flow_from_directory(
    'datasets/images',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# ---------------------------------------------------
# CNN MODEL
# ---------------------------------------------------

model = Sequential([

    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(128,128,3)
    ),

    MaxPooling2D(2,2),

    Conv2D(
        64,
        (3,3),
        activation='relu'
    ),

    MaxPooling2D(2,2),

    Conv2D(
        128,
        (3,3),
        activation='relu'
    ),

    MaxPooling2D(2,2),

    Flatten(),

    Dense(128, activation='relu'),

    Dropout(0.5),

    Dense(1, activation='sigmoid')
])

# ---------------------------------------------------
# COMPILE MODEL
# ---------------------------------------------------

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------

model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# ---------------------------------------------------
# CREATE MODELS FOLDER
# ---------------------------------------------------

if not os.path.exists('models'):
    os.makedirs('models')

# ---------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------

model.save('models/deepfake_cnn.h5')

print("Deepfake CNN Model Saved")