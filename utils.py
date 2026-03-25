import pandas as pd
import numpy as np

def load_data(file):
    return pd.read_csv(file)

def split_multiselect(series):
    return series.fillna("").apply(lambda x: x.split("|"))

def one_hot_encode_multiselect(df, column):
    df = df.copy()
    s = df[column].fillna("")
    unique_items = set()
    for items in s:
        for it in str(items).split("|"):
            if it.strip():
                unique_items.add(it.strip())
    for it in sorted(unique_items):
        df[f"{column}__{it}"] = s.apply(lambda x: 1 if it in str(x).split("|") else 0)
    return df.drop(columns=[column])

def preprocess(df):
    df = df.copy()
    # Handle multi-select columns
    for col in ["Categories","Stress_Purchases","Shopping_Situations","Product_Combinations"]:
        if col in df.columns:
            df = one_hot_encode_multiselect(df, col)

    # Target encoding placeholder (keep as label)
    return df

def encode_for_model(df, target_col=None):
    df = df.copy()
    y = None
    if target_col and target_col in df.columns:
        y = df[target_col]
        df = df.drop(columns=[target_col])

    X = pd.get_dummies(df, drop_first=True)
    return X, y
