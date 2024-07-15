# Detect-Pixelated-Image-And-Correct-It
# Objective:
The primary objective of this project is to develop an AI-based solution to detect and correct pixelation in images. The solution includes a pre-trained classifier to identify pixelated images and uses a Super-Resolution Convolutional Neural Network (SRCNN) to enhance and correct the pixelation.

# Problem Description:
In the field of image processing, pixelation is a common issue that degrades the quality of images, making them less useful for analysis and display. The challenge is to detect pixelation in images and apply appropriate techniques to correct it, thereby restoring the image to a higher quality. This project aims to build an AI solution that detects pixelation using a machine learning classifier and corrects it using an SRCNN model. The solution processes the image, identifies if it is pixelated, and then enhances the image if necessary.

# Solution Features:

# Data Pre-processing:

Convert images to grayscale for feature extraction. Extract Local Binary Patterns (LBP) and edge histograms for classification. 
# Knowledge Representation: 
Represent image features using histograms (LBP and edge histograms). Use visual representations (corrected images) to demonstrate pixelation correction.

# Pattern Identification:

Identify pixelation patterns using a pre-trained classifier.
Detect pixelation based on extracted features from the images.
Insight Generation:

Generate insights by classifying images as pixelated or non-pixelated.
Enhance pixelated images using an SRCNN model and provide the corrected output.
Scalability:

The solution can handle various image sizes and complexities.
The model can be retrained with additional data to improve performance.

User-friendly Interface:

The solution includes clear instructions and paths to input and output images.
Print statements and saved images provide feedback and results to the user.
