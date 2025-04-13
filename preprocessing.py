import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

# Load the dataset
df = pd.read_csv('dataset/fraudTrain.csv')

# Drop irrelevant or non-numeric columns
df = df.drop([
    'Unnamed: 0', 'trans_date_trans_time', 'merchant', 'first', 'last',
    'street', 'city', 'state', 'zip', 'job', 'dob', 'trans_num'
], axis=1)

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=['category', 'gender'], drop_first=True)

# Separate features and label
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance using under-sampling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_scaled, y)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

selected_features = X.columns.tolist()


# Output results
print("\n✅ Preprocessing complete!")
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("Fraud ratio in training set:", sum(y_train)/len(y_train))
