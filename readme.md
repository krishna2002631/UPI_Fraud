# 💳 UPI Fraud Detection System

## Overview

This is a machine learning-powered Streamlit application designed to detect potential fraudulent UPI (Unified Payments Interface) transactions in real-time. The system uses a Random Forest Classifier to assess the likelihood of fraud based on multiple transaction features.

## 🌟 Features

- **Real-time Fraud Detection**
  - Analyze individual transactions for fraud risk
  - Provide instant fraud probability assessment
  - Support for multiple geographical and transactional inputs

- **Machine Learning Model**
  - Random Forest Classifier
  - Automatic model training and evaluation
  - Balanced class weighting for accurate predictions

- **Interactive Dashboard**
  - Real-time fraud detection interface
  - Model performance metrics
  - Risk visualization charts

## 🛠 Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Steps
1. Clone the repository
```bash
git clone https://github.com/yourusername/upi-fraud-detection.git
cd upi-fraud-detection
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install required dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Running the Application

```bash
streamlit run app.py
```

## 📊 How It Works

### Fraud Detection Methodology
The system evaluates transactions based on multiple features:
- Transaction amount
- Sender and receiver transaction history
- Time of day
- Geographical distance
- Device risk score

### Feature Engineering
- Calculates geo-distance between sender and receiver
- Generates time-based features
- Assesses device risk

### Model Training
- Uses Random Forest Classifier
- Employs stratified sampling
- Implements class balancing techniques

## 🔍 Dashboard Sections

### 1. Fraud Detection
- Input transaction details
- Get real-time fraud probability
- Receive fraud alerts

### 2. Model Performance
- Retrain model on new data
- View precision, recall, and F1-score

### 3. Risk Visualization
- Transaction amount distribution
- Fraud occurrences by time of day

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ⚠️ Disclaimer

This is a demonstration project. While it implements machine learning techniques for fraud detection, it should not be used as the sole method of fraud prevention in a production environment.



