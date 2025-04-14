
import pandas as pd
from src.data_loader import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path='data/Churn_Modelling.csv'):
    df = load_data(file_path)

    X = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
    y = df['Exited']

    # One-hot encode categorical variables
    X = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True)

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
