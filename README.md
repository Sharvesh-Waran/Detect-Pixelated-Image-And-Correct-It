<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Image Pixelation Detection and Correction</title>
</head>
<body>
    <header>
        <h1>Image Pixelation Detection and Correction</h1>
    </header>

    <section>
        <h2>Objective</h2>
        <p>
            Develop an AI-based solution to effectively detect and correct pixelation in images. This solution involves image classification to detect pixelation and applying a Super-Resolution Convolutional Neural Network (SRCNN) model to enhance image quality, with applications in autonomous driving, image classification, and image generation.
        </p>
    </section>

    <section>
        <h2>Problem Description</h2>
        <p>
            Pixelation degrades image quality, making images less useful for various applications. This project addresses the challenge of detecting pixelation and enhancing image quality using AI techniques. The goal is to create a system that accurately classifies images as pixelated or non-pixelated and applies a super-resolution model to enhance pixelated images.
        </p>
    </section>

    <section>
        <h2>Solution Features</h2>
        <ul>
            <li><strong>Data Pre-processing:</strong> Convert images to grayscale and apply local binary pattern (LBP) to extract texture features. Use edge detection (Canny edge detector) to extract edge histogram features.</li>
            <li><strong>Knowledge Representation:</strong> Represent extracted features using histograms for LBP and edge histograms. Use a trained classifier to predict if an image is pixelated based on these features.</li>
            <li><strong>Pattern Identification:</strong> Identify pixelation in images using the trained classifier.</li>
            <li><strong>Insight Generation:</strong> Enhance the image quality using the SRCNN model if an image is classified as pixelated. The SRCNN model processes the luminance channel of the image to reconstruct a high-resolution version.</li>
            <li><strong>Scalability:</strong> Handle images of varying sizes and complexities by adjusting the input dimensions for the SRCNN model.</li>
            <li><strong>User-friendly Interface:</strong> Provide a straightforward interface for loading images, detecting pixelation, and saving enhanced images.</li>
        </ul>
    </section>

    <section>
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
                <p>This will build and train the SRCNN model and save the weights to the specified path.</p>
            </li>
            <li>Detect Pixelation in an Image:
                <pre><code>python detect_pixelation.py --image-path path/to/image</code></pre>
                <p>This will load the image and predict whether it is pixelated.</p>
            </li>
            <li>Correct Pixelation in an Image:
                <pre><code>python correct_pixelation.py --image-path path/to/image --output-path path/to/save/corrected/image</code></pre>
                <p>This will load the image, correct pixelation if detected, and save the corrected image to the specified path.</p>
            </li>
        </ol>
    </section>

    <section>
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
