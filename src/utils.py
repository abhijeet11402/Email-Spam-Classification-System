# src/utils.py
import pandas as pd
from typing import Tuple

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load CSV and ensure columns exist. Returns DataFrame with columns 'text' and 'label_num (numeric label (0 = ham, 1 = spam))'.
    If the CSV uses alternate names:
        - Automatically renames 'message', 'body', or 'content' → 'text'
        - Converts 'label' to 'label_num' if needed
    """
    df = pd.read_csv(path)
    # common alternate column names handling
    if 'text' not in df.columns:
        # try some common names
        for alt in ['message', 'body', 'content']:
             # If alternate column exists, rename it to 'text'
             if alt in df.columns:
                df = df.rename(columns={alt: 'text'})
                break
    # Ensure label column exists and convert if necessary
    if 'label_num' not in df.columns:
        if 'label' in df.columns:
            # If labels are strings ('ham', 'spam'), convert to numeric
            if df['label'].dtype == object:
                df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
            # If label is already numeric, rename it
            else:
                df = df.rename(columns={'label': 'label_num'})
        else:
            # Raise error if required columns are missing
            raise ValueError("Dataset must contain 'text' and 'label_num' (or 'label') columns.")
    # Remove rows with missing text values
    df = df.dropna(subset=['text'])
    return df

def save_joblib(obj, path: str):
    import joblib
    joblib.dump(obj, path)
