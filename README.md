# Smile_detection

The "Smile Detection" repository is designed for real-time smile detection using a deep learning model called LeNet. This repository encompasses the development, training, and evaluation of LeNet, a deep neural network, which is trained on a dataset of smiles. After the training phase, the model is utilized to make predictions on a set of test data that comes with ground truth labels. In addition to smile detection, this repository leverages the DeepFace framework to detect and assess the accuracy of recognizing happy emotions within the same test data. This combined approach not only evaluates the model's performance in detecting smiles but also measures its ability to identify happiness in real-time scenarios.

# Key Features
LeNet Smile Detection: LeNet, a deep learning model, is employed for real-time smile detection.
DeepFace Smile Detection: provides a high-level interface for facial analysis tasks, including facial recognition, facial attribute analysis, emotion recognition, and more.
The "DeepFace" library simplifies complex facial analysis tasks by providing pre-trained deep learning models that can be easily used to perform tasks like face recognition and emotion detection. 

Training and Testing: The repository includes scripts for training LeNet on a dataset of smiles and performing smile detection on a test dataset.

Emotion Recognition: DeepFace is used to recognize happiness emotions within the test data, providing a comprehensive assessment.

Accuracy Evaluation: The repository calculates the accuracy of both smile detection using LeNet and happiness recognition using DeepFace.

# Getting Started
To get started with smile detection and emotion recognition, follow these steps:

Clone the Repository: Clone this repository to your local machine using git clone.

Dataset Preparation: Ensure you have a dataset of smiles for training and a separate test dataset with ground truth labels.

Training LeNet: Train the LeNet model on your smile dataset using the provided training script.

Testing Smile Detection: Use the trained model to perform smile detection on your test dataset and calculate accuracy.

Happiness Recognition: Employ DeepFace to detect happiness emotions in the same test dataset and assess the accuracy.

Results and Visualization: Review the results and visualizations to understand the model's performance.

# Dependencies
Ensure you have the following dependencies installed:

Python
DeepFace: pip install deepface 
OpenCV
TensorFlow
NumPy

# Results 
LeNet achieved an accuracy of 0.81 for prediction on the validation set. And below, we can see actual results of LeNet predictions in real time. 
![lenet_model_results_weights](https://github.com/nourhenehanana/Smile_detection/assets/93352403/4deea094-c911-4840-a27b-63e59c1839e5)
![notsmiling](https://github.com/nourhenehanana/Smile_detection/assets/93352403/bc9402c2-8272-4df6-9a2d-872256b45dc6)
![smiling](https://github.com/nourhenehanana/Smile_detection/assets/93352403/86e49096-1608-415d-bb27-d1f07a5be576)

