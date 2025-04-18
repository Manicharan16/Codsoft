from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import joblib

def train(X_train, y_train, method="logistic"):
    """
    method: 'logistic', 'nb', or 'svm'
    """
    if method == "nb":
        from sklearn.naive_bayes import MultinomialNB
        clf = MultinomialNB()
    elif method == "svm":
        from sklearn.svm import LinearSVC
        clf = LinearSVC()
    else:
        clf = LogisticRegression(max_iter=1000)

    clf.fit(X_train, y_train)
    return clf

def evaluate(clf, X, y, le):
    y_pred = clf.predict(X)
    print(classification_report(y, y_pred, target_names=le.classes_))

def save(clf, vec, le, model_dir="models"):
    joblib.dump(clf, f"{model_dir}/classifier.joblib")
    joblib.dump(vec, f"{model_dir}/vectorizer.joblib")
    joblib.dump(le,  f"{model_dir}/label_encoder.joblib")
