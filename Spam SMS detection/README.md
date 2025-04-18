Tech Stack

Language: Python

Libraries: scikit-learn, pandas, numpy, joblib

Model Used: Logistic Regression

Feature Extraction: TF-IDF Vectorizer

Project Structure:

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


Dataset

Source: Kaggle - SMS Spam Collection Dataset
Size: 5,572 SMS messages, labeled as spam or ham.

Sample Output

Enter SMS message:
> You have won $1000 cash prize. Claim now!
Prediction: spam



