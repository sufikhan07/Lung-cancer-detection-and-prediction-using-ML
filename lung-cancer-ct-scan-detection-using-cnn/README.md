# 🫁 Lung-Cancer-CT-Scan-Detection-Using-CNN
This project is a deep learning-based application for detecting lung cancer from CT scan images using a Convolutional Neural Network (CNN). It includes a full GUI built with Tkinter, allowing users to upload CT scan images and receive predictions in real time.

### 📁 Dataset Download

You can download the full Lung Cancer dataset used in this project from the link below:

🔗 [Download Lung Cancer Dataset (Google Drive)](https://drive.google.com/file/d/1KQB5f3MJL_jnsesh8OuzffkpBYhTI3SI/view?usp=drive_link)


## 📁 Project Structure

![image](https://github.com/user-attachments/assets/c3b51d5c-a4e0-4f58-a901-05bd2edefaa3)


## ✅ Features
- Image preprocessing with real-time augmentation

- CNN model architecture for binary classification (Cancerous / Non-Cancerous)

- GUI for image upload and prediction

- Displays patient details, uploaded image, prediction result, and confidence

## 🔧 Requirements
###Make sure the following packages are installed:
- pip install tensorflow pillow numpy

## 🧠 How to Train the Model
1. Place your dataset under LungcancerDataSet/Data/ with the following structure:

![image](https://github.com/user-attachments/assets/b3fee6aa-55ba-44b5-a4e3-e8780b34e29e)

2. Run the training script:
   - python train_cnn_model.py
     
This will:
- Train the CNN model on the dataset
- Save the trained model as lung_cancer_cnn_model.keras
- Print final test accuracy

## 🖼️ How to Run the GUI App
1. Ensure the trained model file lung_cancer_cnn_model.keras is present in the project directory.
2. Run the GUI application:
   - python app_gui.py
     
3. The app will open a window:
- Enter patient information
- Upload a CT scan image
- Click Submit to get prediction

## 📊 Model Overview
### The CNN model includes:

- 3 Convolutional layers with BatchNormalization and MaxPooling
- GlobalAveragePooling before fully connected layers
- Dropout regularization
- Sigmoid output layer for binary classification

## 📌 Notes
- Make sure your dataset path is correct in all scripts.
- The current model expects images resized to 224x224.
- The GUI is designed for binary classification only.
- If modifying class names, also update class_labels in app_gui.py.
