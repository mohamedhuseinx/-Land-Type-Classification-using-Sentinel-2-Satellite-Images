# 🛰 Sentinel-2 Land Type Classification

## 📌 Overview

This project focuses on **Land Use and Land Cover Classification** using satellite images from Sentinel-2.
The goal is to build a **Deep Learning model** that can classify land types (e.g., agricultural, forest, water, urban) from satellite imagery.

---

## 🎯 Objectives

* Build a **CNN model** for satellite image classification
* Improve performance using **Data Augmentation**
* Compare **Custom CNN** vs **Transfer Learning models**
* Deploy the model through an **API**

---

## 📂 Dataset

The project uses the **EuroSAT (RGB version)** dataset:

* ~27,000 images
* 10 classes
* Image size: 64×64 (RGB)

### Classes:

* AnnualCrop
* Forest
* HerbaceousVegetation
* Highway
* Industrial
* Pasture
* PermanentCrop
* Residential
* River
* SeaLake

---

## 🧠 Methodology

### 1. Data Exploration (EDA)

* Analyze class distribution
* Verify dataset balance
* Visualize sample images

### 2. Data Preprocessing

* Resize images to 224×224
* Normalize pixel values to [0, 1]
* Train/Validation split (80/20)
* Apply Data Augmentation (training set only)

### 3. Model Building

* Custom CNN architecture
* Transfer Learning (ResNet / EfficientNet) *(planned)*

### 4. Evaluation

* Accuracy
* Loss curves
* Confusion Matrix *(planned)*

### 5. Deployment

* Build an API using FastAPI
* Upload an image and return predicted class

---

## 📁 Project Structure

```bash
Sentinel2-Land-Classification/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_EDA_Preprocessing.ipynb
│   ├── 02_Model_Building.ipynb
│
├── src/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│
├── models/
│   └── land_classifier.h5
│
├── app/
│   └── main.py
│
├── reports/
│   └── EDA_Report.pdf
│
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Sentinel2-Land-Classification.git
cd Sentinel2-Land-Classification
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run training

```bash
python src/train.py
```

---

## 📊 Results (Coming Soon)

* Model accuracy
* Training curves
* Evaluation metrics

---

## 🔮 Future Work

* Improve performance using Transfer Learning
* Add Confusion Matrix
* Deploy the project to cloud
* Build a dashboard

---

## 👨‍💻 Author

Mohamed Hussein
Ahmed Dino
Mustafa Yunus
Tamer Malek
---

## 📜 License

This project is for educational purposes as part of the DEPI Program.
