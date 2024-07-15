import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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


def prepare_dataset(folder_path, label):
    X = []  
    y = []  

    for filename in os.listdir(folder_path):
        
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            features = extract_features(image_path)
            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)

original_folder_path = r"D:\Intel Unnati Project PS - 9\Dataset\Original" # PATH FOR ORIGINAL DATASET
X_original, y_original = prepare_dataset(original_folder_path, label=0)


pixelated_folder_path = r"D:\Intel Unnati Project PS - 9\Dataset\Pixelated"
X_pixelated, y_pixelated = prepare_dataset(pixelated_folder_path, label=1) # PATH FOR PIXELATED DATESET

X = np.concatenate((X_original, X_pixelated))
y = np.concatenate((y_original, y_pixelated))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)