# Image Pixelation Detection and Correction

## Objective
Develop an AI-based solution to effectively detect and correct pixelation in images. This solution involves image classification to detect pixelation and applying a Super-Resolution Convolutional Neural Network (SRCNN) model to enhance image quality, with applications in autonomous driving, image classification, and image generation.

## Problem Description
Pixelation degrades image quality, making images less useful for various applications. This project addresses the challenge of detecting pixelation and enhancing image quality using AI techniques. The goal is to create a system that accurately classifies images as pixelated or non-pixelated and applies a super-resolution model to enhance pixelated images.

## Solution Features
- **Data Pre-processing:** Convert images to grayscale and apply local binary pattern (LBP) to extract texture features. Use edge detection (Canny edge detector) to extract edge histogram features.
- **Knowledge Representation:** Represent extracted features using histograms for LBP and edge histograms. Use a trained classifier to predict if an image is pixelated based on these features.
- **Pattern Identification:** Identify pixelation in images using the trained classifier.
- **Insight Generation:** Enhance the image quality using the SRCNN model if an image is classified as pixelated. The SRCNN model processes the luminance channel of the image to reconstruct a high-resolution version.
- **Scalability:** Handle images of varying sizes and complexities by adjusting the input dimensions for the SRCNN model.
- **User-friendly Interface:** Provide a straightforward interface for loading images, detecting pixelation, and saving enhanced images.

## How to Work with the Project
1. Clone the Repository:
    ```bash
    git clone <repository-url>
    ```
2. Install Required Libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. Train the SRCNN Model:
    ```bash
    python train_srcnn.py
    ```
    This will build and train the SRCNN model and save the weights to the specified path.
4. Detect Pixelation in an Image:
    ```bash
    python detect_pixelation.py --image-path path/to/image
    ```
    This will load the image and predict whether it is pixelated.
5. Correct Pixelation in an Image:
    ```bash
    python correct_pixelation.py --image-path path/to/image --output-path path/to/save/corrected/image
    ```
    This will load the image, correct pixelation if detected, and save the corrected image to the specified path.

## Made by Bandits
- Sharveshwaran S S
- Jeyakumar S
- Rohith U
- Prithivraj M P
- Ramalingam M
