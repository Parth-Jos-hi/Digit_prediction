👇

🧠 Project Overview

This project is an AI-based Handwritten Digit Recognition Web Application that predicts digits from 0–9 using Deep Learning and Computer Vision. It is powered by a Convolutional Neural Network (CNN) trained on the MNIST dataset, which contains 70,000 grayscale images of handwritten digits. The model learns the spatial structure and visual features of digits such as curves, edges, and loops, achieving an accuracy of around 99%. This allows the system to identify digits even when written with slight variations, making it a robust digit classification solution.

⚙️ Technical Implementation

The model architecture includes two convolutional layers with ReLU activations, followed by max pooling, flattening, and dense layers leading to a final Softmax output layer for classification. The model is built and trained using TensorFlow/Keras and saved in the latest .keras format for improved efficiency and compatibility. The Streamlit framework powers the interactive web interface, enabling users to either draw a digit on a virtual canvas or upload an image. Each input image undergoes preprocessing steps—conversion to grayscale, inversion, resizing to 28×28 pixels, and normalization—to match the model’s input format before prediction.

🚀 Features and Functionality

Once the image is processed, the trained CNN model predicts the corresponding digit and displays the result along with a probability distribution chart showing the model’s confidence for each class. The top three predictions are also displayed for better interpretability. The app uses libraries such as Pillow for image processing, Matplotlib for visualization, and streamlit-drawable-canvas for real-time drawing functionality. Designed to run locally through Streamlit, the project offers a complete end-to-end workflow—from training a deep learning model to deploying it as a functional, interactive web application.
