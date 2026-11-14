"""Reproduces Phase 1 notebook preprocessing without looking at test labels during training."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def load_data(path='creditcard.csv', seed=42):
    df = pd.read_csv(path)
    needed = {'Time', 'Amount', 'Class', *(f'V{i}' for i in range(1, 29))}
    if not needed.issubset(df.columns):
        raise ValueError(f'Missing columns: {sorted(needed - set(df.columns))}')
    if df.isna().any().any():
        raise ValueError('Missing values in input dataset')
    hour = (df['Time'].to_numpy() % 86400) / 3600
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df = df.drop(columns=['Time'])
    X, y = df.drop(columns='Class'), df['Class']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=.30, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=.50, stratify=y_temp, random_state=seed)
    scaler = RobustScaler()  # follows uploaded notebook: includes cyclical features
    arrays = [scaler.fit_transform(X_train), scaler.transform(X_val), scaler.transform(X_test)]
    labels = [y_train.to_numpy(dtype=np.float64), y_val.to_numpy(dtype=np.float64), y_test.to_numpy(dtype=np.float64)]
    if any(a.shape[1] != 31 for a in arrays):
        raise ValueError('Expected 31 features after Phase 1 processing')
    return arrays, labels, list(X.columns)
