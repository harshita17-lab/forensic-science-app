import numpy as np
from tensorflow.keras import layers, models

# ---------------- CNN EMBEDDING MODEL ----------------


def build_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu',
                      input_shape=(128, 128, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128),  # embedding vector (no activation)
    ])

    return model

# ---------------- FEATURE EXTRACTION ----------------


def extract_features(model, img):
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    features = model.predict(img, verbose=0)[0]

    # Normalize embedding (VERY IMPORTANT)
    norm = np.linalg.norm(features)
    if norm == 0:
        return features
    return features / norm
