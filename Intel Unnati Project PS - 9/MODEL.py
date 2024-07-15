import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
import numpy as np
import os

def build_srcnn_model():
    model = Sequential()
    model.add(tf.keras.layers.Input(shape=(None, None, 1)))
    model.add(Conv2D(64, (9, 9), activation='relu', padding='same'))
    model.add(Conv2D(32, (1, 1), activation='relu', padding='same'))
    model.add(Conv2D(1, (5, 5), activation='linear', padding='same'))
    return model

def train_srcnn_model(model):
    # TRAINING DATA
    x_train = np.random.rand(100, 32, 32, 1).astype(np.float32)
    y_train = np.random.rand(100, 32, 32, 1).astype(np.float32)

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=1)  

def save_weights(model, path):
    model.save_weights(path)

if __name__ == "__main__":
    srcnn_model = build_srcnn_model()
    train_srcnn_model(srcnn_model)
    save_path = r'Models/model.weights.h5'  
    save_weights(srcnn_model, save_path)
    print(f"saved successfully to {save_path}.")
