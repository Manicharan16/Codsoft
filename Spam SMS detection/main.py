import os
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocessing import preprocess
from src.model import train, evaluate, save

# 1) Load
df = load_data("data/spam.csv")
print(f"Loaded {len(df)} messages")

# 2) Preprocess
X, y, vec, le = preprocess(df)
print(f"Vectorized into {X.shape[1]} features")

# 3) Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# 4) Train
clf = train(X_train, y_train, method="nb")


# 5) Evaluate
print("Validation Results:")
evaluate(clf, X_val, y_val, le)

# 6) Save
os.makedirs("models", exist_ok=True)
save(clf, vec, le)
print("Model & artifacts saved to /models")
