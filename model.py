import numpy as np
from tensorflow.keras import layers, models


def build_cnn():
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu',
                      input_shape=(128, 128, 1)),
        layers.MaxPooling2D(),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation='relu')
    ])
    return model


def extract_features(model, img):
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return model.predict(img)[0]
