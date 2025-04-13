import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from preprocessing import selected_features  # Import the features used in training

# Load trained model
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

# Example input (replace these values with new transaction data)
new_data = pd.DataFrame([{
    'amt': 1000,
    'city_pop': 50000,
    'unix_time': 1371816927,
    'gender_F': 1,
    'gender_M': 0,
    # Add all other required one-hot encoded or numeric columns used in training
}])

# Ensure the columns are aligned
for col in selected_features:
    if col not in new_data.columns:
        new_data[col] = 0

new_data = new_data[selected_features]

# Scale the input data
scaler = StandardScaler()
new_data_scaled = scaler.fit_transform(new_data)  # For real deployment, use saved scaler

# Predict
prediction = model.predict(new_data_scaled)
result = "Fraudulent Transaction" if prediction[0] == 1 else "Legit Transaction"

print("\nPrediction Result:", result)
