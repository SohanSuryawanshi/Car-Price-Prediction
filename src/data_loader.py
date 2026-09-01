"""
Data loading and train/test splitting.

Split happens here, immediately after loading and dropping unusable columns,
and before any statistic (imputation values, outlier bounds, encoders,
scaler params) is learned anywhere else in the pipeline.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

# Load data
def load_raw_data(path=None)-> pd.DataFrame:
    """Load the raw CSV and drop columns with no predictive value"""

    path = path or config.RAW_DATA_PATH
    df = pd.read_csv(path)

    cols_to_drop = [col for col in config.DROP_COLS if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Also drop rows with a missing target, can't train/evaluate on those
    df = df.dropna(subset=[config.TARGET])
    return df

def split_data(df: pd.DataFrame):
    """Split into train/test. Do this before fitting anything else."""
    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")