# 🌿 Plant Disease Image Classifier

A Convolutional Neural Network (CNN) based image classification model built using TensorFlow to detect plant diseases from leaf images. This model helps in identifying whether a plant is healthy or suffering from any common disease.

---

## 📌 Project Overview

This project uses Deep Learning —specifically CNNs— to classify images of plant leaves into different categories of diseases (including healthy). It's aimed at helping farmers, researchers, and agricultural extension workers detect diseases early and take action.

---

## 🧠 Technologies Used

- Python
- TensorFlow & Keras
- NumPy & Pandas
- Matplotlib & Seaborn (for visualization)
- Streamlit (for building a simple app interface)

---

## 🗂 Dataset

The dataset consists of thousands of labeled plant leaf images across various classes such as:

- Apple — Healthy, Apple Scab, Black rot, Cedar Apple Rust  
- Cherry (including sour) — Healthy, Powdery mildew  
- Corn — Healthy, Common Rust, Northern Leaf Blight  

📦 **Source:** *[[https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset]]*  
Images are organized into folders by class and loaded using Keras' `ImageDataGenerator`.

---

## 🔍 Project Workflow

### 1. Data Preprocessing

- Loaded images using Keras’ `ImageDataGenerator`.
- Applied image rescaling and batching.
- Split data into training and validation sets.

### 2. Model Architecture

- A custom CNN with Conv2D, MaxPooling, Dropout, and Dense layers.
- Used ReLU activations and Softmax for classification.

### 3. Training

- Loss Function: Categorical Crossentropy  
- Optimizer: Adam  
- Evaluation metrics: Accuracy  

### 4. Evaluation

- Accuracy and loss curves plotted.

### 5. Streamlit App

- Built a simple app with Streamlit in `main.py`.
- Users can upload a plant leaf image and receive a model prediction.

---

## 🚀 How to Run

1. **Clone the repo:**

```
git clone https://github.com/yourusername/plant-disease-image-classification.git
cd plant-disease-image-classification
```

2. **Install dependencies**

```
pip install -r requirements.txt
```

3. **Launch the Streamlit app**

```
streamlit run main.py
```
The app will open in your default browser. Upload a plant leaf image and view the prediction.

---

## 🧪 Results

- Achieved 83% validation accuracy.
- Correctly identifies multiple plant diseases with good performance on unseen data.

![App Screenshot]

---

## 🛠 Future Improvements

- Increase model accuracy through deeper or pre-trained architectures.
- Apply **data augmentation** to improve generalization.
- Use **transfer learning** (e.g., ResNet, EfficientNet).
- Add **cross-validation** to improve robustness.
- Deploy the app to the web using platforms like **Streamlit Cloud** or **Render**.

---

