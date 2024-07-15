<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Pixelation Detection and Correction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
            color: #333;
        }
        header {
            background-color: #333;
            color: #fff;
            padding: 20px 0;
            text-align: center;
        }
        header h1 {
            margin: 0;
        }
        section {
            padding: 20px;
            margin: 20px;
            background: #fff;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        h2 {
            border-bottom: 2px solid #333;
            padding-bottom: 5px;
            margin-bottom: 10px;
        }
        ul {
            list-style-type: disc;
            padding-left: 20px;
        }
        ol {
            list-style-type: decimal;
            padding-left: 20px;
        }
        pre {
            background: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow: auto;
        }
        code {
            font-family: 'Courier New', Courier, monospace;
            background: #eee;
            padding: 2px 4px;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <header>
        <h1>Image Pixelation Detection and Correction</h1>
    </header>
    <section id="objective">
        <h2>Objective</h2>
        <p>The primary objective of this project is to develop an AI-based solution that can effectively detect and correct pixelation in images. The solution involves multiple steps, including image classification to detect pixelation and applying a Super-Resolution Convolutional Neural Network (SRCNN) model to enhance the image quality. This solution aims to improve image quality, particularly in the context of autonomous driving, image classification, and image generation.</p>
    </section>
    <section id="problem-description">
        <h2>Problem Description</h2>
        <p>Pixelation can significantly degrade the quality of images, making them less useful for various applications. This project addresses the challenge of detecting pixelation in images and subsequently enhancing the image quality using AI techniques. The goal is to create a system that can accurately classify images as pixelated or non-pixelated and apply a super-resolution model to enhance pixelated images.</p>
    </section>
    <section id="features">
        <h2>Solution Features</h2>
        <ul>
            <li><strong>Data Pre-processing:</strong> Convert images to grayscale and apply local binary pattern (LBP) to extract texture features. Use edge detection (Canny edge detector) to extract edge histogram features.</li>
            <li><strong>Knowledge Representation:</strong> Represent extracted features in the form of histograms for LBP and edge histograms. Use a trained classifier to predict if an image is pixelated or not based on the extracted features.</li>
            <li><strong>Pattern Identification:</strong> Identify pixelation in images using the trained classifier.</li>
            <li><strong>Insight Generation:</strong> If an image is classified as pixelated, enhance the image quality using the SRCNN model. The SRCNN model processes the luminance channel of the image and reconstructs a high-resolution version.</li>
            <li><strong>Scalability:</strong> The solution can handle images of varying sizes and complexities by adjusting the input dimensions for the SRCNN model.</li>
            <li><strong>User-friendly Interface:</strong> The solution provides a straightforward interface for loading images, detecting pixelation, and saving enhanced images.</li>
        </ul>
    </section>
    <section id="how-to-work">
        <h2>How to Work with the Project</h2>
        <ol>
            <li>Clone the Repository:
                <pre><code>git clone &lt;repository-url&gt;</code></pre>
            </li>
            <li>Install Required Libraries:
                <pre><code>pip install -r requirements.txt</code></pre>
            </li>
            <li>Train the SRCNN Model:
                <pre><code>python train_srcnn.py</code></pre>
                <p>This will build and train the SRCNN model, then save the weights to the specified path.</p>
            </li>
            <li>Detect Pixelation in an Image:
                <pre><code>python detect_pixelation.py --image-path path/to/image</code></pre>
                <p>This will load the image and predict whether it is pixelated or not.</p>
            </li>
            <li>Correct Pixelation in an Image:
                <pre><code>python correct_pixelation.py --image-path path/to/image --output-path path/to/save/corrected/image</code></pre>
                <p>This will load the image, correct the pixelation if detected, and save the corrected image to the specified path.</p>
            </li>
        </ol>
    </section>
    <section id="team">
        <h2>Made by Bandits</h2>
        <ul>
            <li>Sharveshwaran S S</li>
            <li>Jeyakumar S</li>
            <li>Rohith U</li>
            <li>Prithivraj M P</li>
            <li>Ramalingam M</li>
        </ul>
    </section>
</body>
</html>
