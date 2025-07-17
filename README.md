🧠 CNN-Based Image Classification on CIFAR-10 Dataset

This project demonstrates how to build a Convolutional Neural Network (CNN) from scratch using TensorFlow/Keras to classify images from the CIFAR-10 dataset. The model was trained and evaluated on 10 categories of real-world images such as airplanes, automobiles, birds, cats, and more.

📌 Project Overview

🧑 Author: Chanuga Jayarathne

🧪 Model Type: Custom CNN (no transfer learning)

📚 Dataset: CIFAR-10 (60,000 32x32 color images in 10 classes)

🧠 Task: Multiclass image classification

⚙️ Frameworks: Python, TensorFlow, Keras, NumPy, Matplotlib

📁 CIFAR-10 Dataset Classes

Airplane

Automobile

Bird

Cat

Deer

Dog

Frog

Horse

Ship

Truck

🏗️ Model Architecture

text
Copy
Edit
Input Layer: 32x32x3
→ Conv2D → ReLU → MaxPooling2D → BatchNorm
→ Conv2D → ReLU → MaxPooling2D → BatchNorm
→ Flatten
→ Dense Layer (128) → ReLU
→ Dense Layer (10) → Softmax (Output)

🧼 Preprocessing Steps

Normalized pixel values to range [0, 1]

One-hot encoded the target labels

Data Augmentation applied to training data to reduce overfitting

🎯 Evaluation Metrics

Accuracy

Confusion Matrix

Classification Report (Precision, Recall, F1-Score)

📊 Results

Training Accuracy: ~70%

Testing Accuracy: ~70%

The model performs well on simpler classes (e.g., ship, airplane), but moderately on visually similar ones (e.g., cat vs. dog).

📈 Loss & Accuracy Curves

Training and validation performance over 20 epochs were plotted to observe overfitting and generalization.

🔍 Future Improvements

Use Transfer Learning (e.g., ResNet, VGG16) to boost accuracy

Apply hyperparameter tuning using KerasTuner or GridSearchCV

Try deeper models or ensemble techniques

🚀 How to Run

bash
Copy
Edit
pip install -r requirements.txt
python cnn_cifar10.py

📬 Contact

For questions or collaboration:
📧 chanugajay@gmail.com
