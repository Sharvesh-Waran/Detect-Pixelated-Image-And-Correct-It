    # DETECT THE IMAGE IS PIXELATED OR NOT
# DETECTION MODEL
import joblib
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

classifier = joblib.load(r"Models/model.joblib") # MODEL PATH
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

new_image_path = r"D:\Intel Unnati Project PS - 9\Dataset\Pixelated\Nature (1).jpg" 
# IMAGE PATH TO FIND PIXELATED IMAGE OR NOT
predicted_class = predict_image_class(new_image_path)
print("Predicted class:", "Pixelated" if predicted_class == 1 else "Non-Pixelated")
