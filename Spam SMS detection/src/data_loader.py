import pandas as pd

def load_data(path="data/spam.csv"):
    """
    Returns DataFrame with columns ['label', 'message'].
    """
    df = pd.read_csv(path, encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
    df.dropna(inplace=True)
    return df
