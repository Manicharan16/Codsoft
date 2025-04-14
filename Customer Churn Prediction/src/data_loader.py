
import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded from {file_path}")
        return df
    except FileNotFoundError:
        print(f"File not found at {file_path}")
        return None
