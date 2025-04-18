# SMS Spam Detection

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Model Accuracy](https://img.shields.io/badge/Accuracy-95%25-brightgreen)](#-model-performance)
[![Made with 💖](https://img.shields.io/badge/Made%20with-%F0%9F%92%96-red)](#)

An AI-based machine learning project that classifies SMS messages as **spam** or **ham (legitimate)**. This project uses Natural Language Processing (NLP) techniques such as **TF-IDF** and a **Logistic Regression classifier** to accurately detect unwanted messages.


## Features

-  Clean and structured codebase using modular Python scripts
-  Preprocessing using TF-IDF vectorizer
-  Trained model using Logistic Regression (high accuracy)
-  Command-line interface (CLI) for real-time message prediction
-  Easy to extend and deploy


## Project Structure

SMS spam detection/
│
├── data/
│   └── spam.csv                         # Original dataset
│
├── models/
│   ├── tfidf_vectorizer.joblib             # Saved TF-IDF vectorizer
│   └── spam_classifier.joblib              # Trained ML model
│
├── src/
│   ├── preprocessing.py                 # Data loading & processing
│   └── model.py                         # Training & evaluation logic
│
├── main.py                              # Main script to train & save model
├── predict.py                           # CLI for user to test message prediction
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation


## Techniques Used

- **Natural Language Processing (NLP)**
- **TF-IDF (Term Frequency-Inverse Document Frequency)**
- **Logistic Regression Classifier**
- **Model persistence using joblib**


## Dataset

- **Name:** SMS Spam Collection
- **Source:** [UCI / Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Size:** 5,572 SMS messages
- **Labels:** `spam`, `ham`


## Model Performance

| Metric       | Score |
|--------------|-------|
| Accuracy     | 95%   |
| Precision    | 95%   |
| Recall       | 95%   |
| F1 Score     | 94%   |



## Sample Output

Enter SMS message:
> You have won $1000 cash prize. Claim now!
Prediction: spam



