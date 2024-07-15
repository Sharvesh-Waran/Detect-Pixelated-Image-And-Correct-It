import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
import joblib
from skimage.feature import local_binary_pattern
import torch

# Function to build the SRCNN model
def build_srcnn_model():
    model = Sequential()
    model.add(tf.keras.layers.Input(shape=(None, None, 1)))
    model.add(Conv2D(64, (9, 9), activation='relu', padding='same'))
    model.add(Conv2D(32, (1, 1), activation='relu', padding='same'))
    model.add(Conv2D(1, (5, 5), activation='linear', padding='same'))
    return model

# Load the classifier and SRCNN model
classifier = joblib.load("Models/model.joblib")  # PATH
srcnn = build_srcnn_model()
srcnn.load_weights('Models/model.weights.h5')  # PATH

def extract_lbp_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_edge_histogram_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    hist, _ = np.histogram(edges.ravel(), bins=np.arange(0, 361, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_features(image_path):
    image = cv2.imread(image_path)
    lbp_features = extract_lbp_features(image)
    edge_histogram_features = extract_edge_histogram_features(image)
    features = np.concatenate((lbp_features, edge_histogram_features))
    return features

def predict_image_class(image_path):
    features = extract_features(image_path)
    prediction = classifier.predict([features])
    return prediction[0]

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(image)
    y = y.astype(np.float32) / 255.0
    y = np.expand_dims(y, axis=0)
    y = np.expand_dims(y, axis=-1)
    return y, cr, cb

def postprocess_image(y, cr, cb):
    y = np.squeeze(y) * 255.0
    y = y.clip(0, 255)
    y = y.astype(np.uint8)
    image = cv2.merge([y, cr, cb])
    image = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
    return image

def correct_pixelation(image_path, model):
    image = cv2.imread(image_path)
    y, cr, cb = preprocess_image(image)
    y_sr = model.predict(y)
    corrected_image = postprocess_image(y_sr, cr, cb)
    return corrected_image

def save_image(image, path):
    cv2.imwrite(path, image)


input_image_path = r"Dataset/Pixelated/Nature (1).jpg"  # INPUT IMAGE PATH
output_image_path = r"Output/1_corrected.png"  # OUTPUT PATH
predicted_class = predict_image_class(input_image_path)
print("Predicted class:", "Pixelated" if predicted_class == 1 else "Non-Pixelated")

if predicted_class == 1:
    corrected_image = correct_pixelation(input_image_path, srcnn)
    save_image(corrected_image, output_image_path)
    print(f"Corrected image saved to {output_image_path}")
