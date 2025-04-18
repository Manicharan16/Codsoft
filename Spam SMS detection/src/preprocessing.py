from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def preprocess(df):
    """
    - Encodes 'spam'/'ham' to 1/0
    - Fits TF-IDF on messages
    Returns: X, y, vectorizer, label_encoder
    """
    # Label encode
    le = LabelEncoder()
    y = le.fit_transform(df['label'])  # spam=1, ham=0

    # TF-IDF vectorization
    vec = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vec.fit_transform(df['message'])

    return X, y, vec, le
