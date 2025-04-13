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

# 1. Overview of the dataset's data types and basic stats
print("\nData Types:")
print(df.dtypes)

print("\nSummary Statistics:")
print(df.describe())

# 2. Visualizing transaction amounts by class
plt.figure(figsize=(10, 5))
sns.boxplot(x='is_fraud', y='amt', data=df)
plt.title('Transaction Amount Distribution by Class')
plt.xticks([0, 1], ['Legit (0)', 'Fraud (1)'])
plt.xlabel("Transaction Type")
plt.ylabel("Amount")
plt.show()

# 3. Check correlation between numerical features
correlation = df.corr(numeric_only=True)

# Plotting the heatmap of correlations
plt.figure(figsize=(12, 8))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Feature Correlation Matrix')
plt.show()

# 4. Most common merchants in fraudulent transactions
top_merchants = df[df['is_fraud'] == 1]['merchant'].value_counts().head(10)
print("\nTop 10 Merchants Involved in Fraudulent Transactions:")
print(top_merchants)

# 5. Visualize frequency of fraudulent transactions by hour
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['hour'] = df['trans_date_trans_time'].dt.hour

plt.figure(figsize=(10, 5))
sns.countplot(x='hour', data=df[df['is_fraud'] == 1], palette='magma')
plt.title('Fraudulent Transactions by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Fraudulent Transactions')
plt.show()
