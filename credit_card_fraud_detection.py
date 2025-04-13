# Import necessary libraries
import pandas as pd
import numpy as np

# Data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Load the train and test datasets
train_df = pd.read_csv('dataset/fraudTrain.csv')
test_df = pd.read_csv('dataset/fraudTest.csv')

# Combine both datasets for a unified analysis
df = pd.concat([train_df, test_df], ignore_index=True)

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Get the shape of the dataset
print("\nDataset Shape:", df.shape)

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Check the distribution of fraud vs legit transactions
print("\nClass distribution (fraud vs legitimate transactions):")
print(df['is_fraud'].value_counts())

# Visualize class distribution
sns.countplot(x='is_fraud', data=df)
plt.title('Class Distribution')
plt.xticks([0, 1], ['Legit (0)', 'Fraud (1)'])
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.show()
