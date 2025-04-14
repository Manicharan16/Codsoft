
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    print("Performing EDA...")

    # Ensure the 'outputs' directory exists
    os.makedirs("outputs", exist_ok=True)

    # Existing EDA code
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nMissing values:\n", df.isnull().sum())

    # Save plot
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Exited', data=df)
    plt.title("Churn Distribution")
    plt.savefig("outputs/churn_distribution.png")
    plt.close()
