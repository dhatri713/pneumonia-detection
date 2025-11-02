# 🫁 Pneumonia Detection using Chest X-Ray Images

An AI-powered web application for detecting pneumonia from chest X-ray images using deep learning. This project compares multiple state-of-the-art CNN architectures and deploys the best-performing model through a FastAPI backend with an intuitive web interface.

**Key Highlights:**
- 🏆 Compared 3+ deep learning architectures
- 🎯 Custom CNN achieved best performance
- 🚀 Real-time inference through FastAPI
- 🌐 User-friendly web interface
- 📊 Comprehensive evaluation metrics

## ✨ Features

- **Multiple Architecture Support**: Experimented with MobileNetV2, EfficientNetB0, and Custom CNN
- **Optimized Threshold**: JSON-based threshold configuration for optimal precision-recall balance
- **REST API**: FastAPI-based backend for easy integration
- **Interactive UI**: Clean HTML/CSS/JavaScript frontend
- **Real-time Predictions**: Fast inference on uploaded X-ray images
- **Confidence Scores**: Probability-based predictions with confidence levels

## 📁 Project Structure

```
lung-chest-xray-classifier/
├── notebooks/
│   ├── custom_cnn.ipynb           # Best performing model ⭐
│   ├── mobilenetv2.ipynb          # Transfer learning with MobileNetV2
│   └── efficientnetb0.ipynb       # Transfer learning with EfficientNetB0
├── app.py                         # FastAPI backend application
├── index.html                     # Frontend web interface
├── lung_cnn_inference.keras       # Trained model (generated after training)
├── lung_cnn_threshold.json        # Optimal threshold (generated after training)
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## 🚀 Setup
1. Create a virtual environment: python -m venv venv
2. pip install --upgrade pip
3. Install all the libraries required: pip install -r requirements.txt
4. Open custome_cnn.ipynb notebook and run it till the last cell. Model related files will be saved in the same directory
5. Start backend: uvicorn app:app --host 0.0.0.0 --port 8000
6. Start the front end: python -m http.server 8080
7. Then open your browser and navigate to:
```
http://localhost:8080
```

### Using the Application

1. **Upload Image**: Click "Choose File" and select a chest X-ray image (JPG, PNG)
2. **Analyze**: Click "Analyze X-Ray"
3. **View Results**: The app will display:
   - Prediction (Normal/Pneumonia)
   - Confidence percentage
   - Visual indicators