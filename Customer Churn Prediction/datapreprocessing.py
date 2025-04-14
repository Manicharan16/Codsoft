# datapreprocessing.py
from data_loader import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    # Load the dataset
    df = load_data()  # No need to pass file path as it is handled by load_data() function

    # Handle missing values if any (based on exploration)
    df = df.dropna()

    # Features (X) and target (y)
    X = df.drop(columns=['Exited', 'RowNumber', 'CustomerId', 'Surname'])
    y = df['Exited']

    # One-hot encode categorical columns like Geography and Gender
    X = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True)

    # Split the data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()
    print("Data preprocessing complete")
