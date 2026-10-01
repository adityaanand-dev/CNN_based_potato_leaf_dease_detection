# 🥔 Potato Leaf Disease Detection System

An AI-driven diagnostic system for detecting and classifying potato leaf diseases (*Early Blight*, *Late Blight*, and *Healthy*) using Deep Learning and Explainable AI (Grad-CAM).

Available via both a **Streamlit Web Dashboard** and a **Flask REST Web Application**.

---

## 🌟 Key Features

1. **Multi-Model Consensus & Comparison**:
   - Compares predictions and confidence scores across three distinct architectures simultaneously:
     - **Custom CNN**: Lightweight sequential architecture (~1.07 MB) with ~95.7% test accuracy.
     - **InceptionV3**: Transfer learning model with ~98.0% test accuracy.
     - **ResNet50**: Transfer learning model with fine-tuned residual layers (~88.0% test accuracy).

2. **Explainable AI (Grad-CAM Visualizations)**:
   - Uses Gradient-weighted Class Activation Mapping to project activation heatmaps onto leaf images.
   - Allows users to visually verify which spots, lesions, or leaf areas triggered the model's diagnosis.
   - Supports selecting any model (InceptionV3, Custom CNN, ResNet50) for Grad-CAM inspection.

3. **Actionable Agricultural Remedies**:
   - Displays cause, visual symptoms, cultural controls, and chemical/biological fungicide treatments for detected diseases.

4. **Live Webcam & Real-Time Video Stream**:
   - Single-frame snapshot mode using webcam.
   - Continuous real-time video stream with 15-frame rolling consensus diagnosis and dynamic lesion bounding boxes.

---

## 📊 Model Performance Benchmarks

Evaluated on the independent test set (300 potato leaf images: 100 Early Blight, 100 Late Blight, 100 Healthy):

| Architecture | Model Size | Test Accuracy | Early Blight F1 | Late Blight F1 | Healthy F1 | Latency (CPU) |
|---|---|---|---|---|---|---|
| **InceptionV3** | 91.2 MB | **~98.0%** | **0.99** | **0.98** | **0.99** | ~120 ms |
| **Custom CNN** | **1.07 MB** | **~95.7%** | **0.95** | **0.94** | **0.98** | **~45 ms** |
| **ResNet50** | 95.9 MB | **~88.0%** | 0.93 | 0.85 | 0.86 | ~160 ms |

> **Recommendation**: 
> - **InceptionV3** delivers the highest diagnostic reliability and precision across all disease categories.
> - **Custom CNN** is ideal for resource-constrained, mobile, or edge deployments (100x smaller footprint with 95.7% accuracy).

---

## 🚀 Getting Started

### 1. Requirements
Ensure the required dependencies are installed:
```bash
pip install streamlit flask tensorflow opencv-python-headless pillow numpy matplotlib
```

### 2. Run the Streamlit Application
```bash
cd StreamlitApp
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Run the Flask Application
```bash
cd FlaskApp
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

### 4. Run the Model Verification Suite
To run inference benchmarks and verify Grad-CAM across models on test images:
```bash
python verify_model.py
```

---

## 📁 Project Structure

```
├── PotatoLeafDiseaseDetection-master/
│   ├── Potato/                      # Dataset (Train, Test, Valid splits)
│   ├── best_potato_leaf_model.h5    # Best custom checkpoint weights
│   ├── verify_model.py              # Verification & benchmark test suite
│   ├── PotatoLeaf.ipynb             # Research and training notebook
│   ├── StreamlitApp/
│   │   ├── app.py                   # Streamlit interactive application
│   │   ├── Home.py                  # Home landing module
│   │   ├── About.py                 # About & documentation module
│   │   └── model/                   # Model weight binaries (.h5)
│   ├── FlaskApp/
│   │   ├── app.py                   # Flask server and API endpoints
│   │   ├── model/                   # Model weight binaries (.h5)
│   │   ├── templates/               # HTML UI templates (index, results, realtime, about, error)
│   │   └── uploads/                 # Uploaded images and generated Grad-CAM outputs
│   └── README.md
```

---

## 🩺 Supported Disease Classes

1. **Early Blight (*Alternaria solani*)**:
   - Symptoms: Concentric brown/black rings (target board pattern) on older foliage.
   - Recommended Treatment: Copper fungicides, Chlorothalonil, Mancozeb, drip irrigation, removal of affected foliage.

2. **Late Blight (*Phytophthora infestans*)**:
   - Symptoms: Rapidly expanding water-soaked purplish-black lesions with white fungal growth on undersides in humid conditions.
   - Recommended Treatment: Systemic fungicides (Metalaxyl, Cymoxanil), immediate eradication of infected plants, improved drainage.

3. **Healthy**:
   - Symptoms: Uniform green leaf structure with no lesions.
   - Recommended Care: Regular nutrient balancing, crop rotation, and periodic humidity scouting.
