# Face-Recognition-using-VGG16
# Transfer Learning
# Face Recognition Using VGG16 with Keras
This project implements a face recognition system using transfer learning with the VGG16 pre-trained model. The model is fine-tuned for the specific task of classifying images into four categories, based on facial features. The dataset used is organized into training and testing sets with images of faces across different categories.

# Features
1. Transfer Learning: Utilizes the pre-trained VGG16 model from ImageNet to extract useful features.
2. Custom Classification Layer: Adds custom dense layers on top of the VGG16 architecture for face classification.
3. Data Augmentation: The training data is augmented in real-time using transformations such as rescaling, zoom, shear, and horizontal flipping to enhance generalization.
4. Real-Time Training Visualization: Loss and accuracy metrics are plotted after training to visualize performance.
5. Model Saving: Trained model is saved as a .h5 file for future inference

# Project Structure
├── Datasets
│   ├── Train (Contains folders for each class)
│   └── Test  (Contains folders for each class)
├── facefeatures_new_model.h5  (Trained model)
└── face_recognition.py  (Python script for training and evaluation)
# Requirements
Python 3.x
TensorFlow
Keras
NumPy
Matplotlib
You can install the dependencies using:   
-> pip install tensorflow keras numpy matplotlib
# How it Works
1. Data Preprocessing:

-> The images are resized to 224x224 and rescaled by dividing pixel values by 255 to normalize the data.
-> ImageDataGenerator is used for real-time data augmentation to improve the model’s robustness.
2. Model Architecture:

-> The VGG16 model is loaded with pre-trained weights from ImageNet, excluding the top layers (which are specific to the original task of classifying ImageNet categories).
-> A Flatten layer is added to convert the 2D outputs of the VGG16 model into a 1D vector.
-> A Dense layer with a softmax activation is used to classify the images into one of the four categories (as per the number of folders in the training set).
3. Training:

-> The model is compiled using the Adam optimizer and categorical cross-entropy loss function.
-> The model is trained for 5 epochs with the training data and validated on the test data.
-> Accuracy and loss values are plotted to track model performance.
4. Saving the Model:

-> After training, the model is saved as facefeatures_new_model.h5.
# Usage
1. Prepare the Dataset:

-> Place your training and testing images in the appropriate Datasets/Train and Datasets/Test directories.
-> Ensure that each class (face category) has its own folder.
2. Run the Training:

-> Run the face_recognition.py script to train the model:
-> python face_recognition.py
3. Model Output:

-> The trained model will be saved as facefeatures_new_model.h5 for future inference.
# Results
-> Training and validation losses and accuracies are plotted and saved as images during training to visualize performance improvements.
# Future Work
-> Extend the model to handle more classes.
-> Optimize the model by experimenting with different architectures or by fine-tuning more layers.
-> Add inference code to recognize faces in real-time using a webcam.
