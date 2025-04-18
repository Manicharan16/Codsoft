import joblib

def predict_sms(text):
    vec = joblib.load("models/vectorizer.joblib")
    le  = joblib.load("models/label_encoder.joblib")
    clf = joblib.load("models/classifier.joblib")

    X = vec.transform([text])
    pred = clf.predict(X)[0]
    return le.inverse_transform([pred])[0]

if __name__ == "__main__":
    sms = input("Enter SMS message:\n> ")
    print("Prediction:", predict_sms(sms))
