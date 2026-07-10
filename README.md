# 🦷 Oral Cancer Detection Using Deep Learning

A Deep Learning-based web application for **Oral Cancer Detection and Segmentation** using a **U-Net + ResNet50** architecture. The system analyzes oral cavity images, identifies suspicious cancer regions, generates segmentation masks, and provides visual predictions through an interactive web dashboard.

## 🚀 Live Demo

🔗 Hugging Face Deployment:  
https://huggingface.co/spaces/dhinesh66/Oral_Cancer_Detection

Model link:
https://drive.google.com/drive/folders/1IfBBj6e0riTeuOU_GsV6_P7sWqafrF9x?usp=sharing

---

## 📌 Project Overview

Oral cancer is one of the most common and dangerous forms of cancer worldwide. Early detection can significantly improve treatment outcomes.

This project uses a Deep Learning segmentation model to:

- Detect cancerous regions in oral images
- Generate tumor segmentation masks
- Estimate cancer coverage percentage
- Visualize predictions through a user-friendly dashboard
- Allow users to upload custom images for analysis

---

## 🛠 Technologies Used

### Frontend
- HTML5
- CSS3
- JavaScript

### Backend
- Flask

### Deep Learning
- TensorFlow
- Keras
- U-Net Architecture
- ResNet50 Encoder

### Data Processing
- NumPy
- OpenCV
- Pillow

### Visualization
- Matplotlib
- Plotly

---

## 🧠 Model Architecture

The model combines:

### U-Net
- Encoder-Decoder segmentation architecture
- Captures both local and global image features

### ResNet50
- Pretrained backbone network
- Extracts high-level features from oral images

### Output
- Binary Segmentation Mask
- Cancer Region Highlighting
- Cancer Percentage Estimation

---

## 📂 Dataset

Dataset used for training and evaluation:

🔗 Google Drive Dataset Link:  
https://drive.google.com/drive/folders/1XbR3oUBNxg4qsWe2KtS3I_YrZvdLAqBR?usp=sharing

### Dataset Structure

```
dataset/
│
├── cancer/
│   ├── images/
│   └── masks/
│
└── normal/
    ├── images/
    └── masks/
```

### Image Information

- Image Size: 224 × 224
- Image Format: JPG / PNG
- Mask Format: Binary PNG Masks

---

## 📊 Training Details

| Parameter | Value |
|------------|---------|
| Model | U-Net + ResNet50 |
| Input Size | 224 × 224 × 3 |
| Batch Size | 8 |
| Epochs | 30 |
| Optimizer | Adam |
| Loss Function | Dice Loss + Binary Cross Entropy |
| Framework | TensorFlow/Keras |

---

## 📈 Evaluation Metrics

The model performance is evaluated using:

- Dice Coefficient
- Intersection over Union (IoU)
- Accuracy
- Binary Cross Entropy Loss

### Formulae

Dice Score:

Dice = (2TP) / (2TP + FP + FN)

IoU Score:

IoU = TP / (TP + FP + FN)

---

## 🌐 Web Application Features

✅ Upload Oral Images

✅ Automatic Preprocessing

✅ Tumor Segmentation

✅ Cancer Region Detection

✅ Prediction Visualization

✅ Interactive Dashboard

✅ Real-Time Inference

---

## 📷 Application Workflow

1. Upload an oral image.
2. Image is resized and normalized.
3. U-Net + ResNet50 model processes the image.
4. Segmentation mask is generated.
5. Cancer percentage is calculated.
6. Results are displayed on the dashboard.

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/oral-cancer-detection.git
cd oral-cancer-detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## 📁 Project Structure

```
oral_cancer_detection/
│
├── app.py
├── train.py
├── requirements.txt
├── generate_normal_masks.py
│
├── model/
│   ├── unet_model.py
│   └── best_model.h5
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
├── dataset/
│   ├── cancer/
│   └── normal/
│
└── logs/
```

---

## 🎯 Future Enhancements

- Multi-class lesion segmentation
- Mobile Application Deployment
- Explainable AI (XAI)
- Grad-CAM Visualization
- Clinical Report Generation
- Cloud-Based Inference API

---

## 👨‍💻 Author

**Dhinesh R**

College Student | Machine Learning & Deep Learning Enthusiast

---

## 📜 License

This project is developed for educational and research purposes.

---

## ⭐ Acknowledgements

- TensorFlow
- Keras
- Flask
- OpenCV
- U-Net Research Paper
- ResNet Research Paper
- Hugging Face Spaces
