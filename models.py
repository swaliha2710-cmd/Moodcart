import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

import joblib

from mlxtend.frequent_patterns import apriori, association_rules

from utils import preprocess, encode_for_model

RANDOM_STATE = 42

def train_classification(df, target_col="Interest_in_MoodCart"):
    df_p = preprocess(df)
    X, y = encode_for_model(df_p, target_col=target_col)

    # Encode target
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=None),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    results = []
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1] if len(le.classes_) == 2 else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        roc = None
        if y_prob is not None:
            try:
                roc = roc_auc_score(y_test, y_prob)
            except:
                roc = None

        results.append({
            "model": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc
        })
        trained[name] = model

    # pick best by f1
    best_name = sorted(results, key=lambda x: x["f1_score"], reverse=True)[0]["model"]
    best_model = trained[best_name]

    return results, best_model, le, X.columns

def train_regression(df, target_col="Monthly_Spend"):
    df_p = preprocess(df)
    X, y = encode_for_model(df_p, target_col=target_col)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    }

    scores = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        scores[name] = r2
        trained[name] = model

    best_name = max(scores, key=scores.get)
    return scores, trained[best_name]

def train_clustering(df, k=4):
    df_p = preprocess(df)
    X, _ = encode_for_model(df_p, target_col=None)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    return labels, km

def association_mining(df, min_support=0.05):
    # use product combinations for simplicity
    if "Product_Combinations" not in df.columns:
        return pd.DataFrame()

    s = df["Product_Combinations"].fillna("").apply(lambda x: str(x).split("|"))
    unique_items = set([item for sub in s for item in sub if item])

    onehot = pd.DataFrame(0, index=df.index, columns=sorted(unique_items))
    for idx, items in s.items():
        for it in items:
            if it in onehot.columns:
                onehot.loc[idx, it] = 1

    freq = apriori(onehot, min_support=min_support, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=0.3)
    rules = rules.sort_values(by=["confidence","lift"], ascending=False)
    return rules

def save_model(model, le, feature_cols, path_prefix="model"):
    joblib.dump(model, f"{path_prefix}.joblib")
    joblib.dump(le, f"{path_prefix}_labelencoder.joblib")
    joblib.dump(list(feature_cols), f"{path_prefix}_features.joblib")

def load_model(path_prefix="model"):
    model = joblib.load(f"{path_prefix}.joblib")
    le = joblib.load(f"{path_prefix}_labelencoder.joblib")
    feats = joblib.load(f"{path_prefix}_features.joblib")
    return model, le, feats

def predict_new(df_new, model, le, feature_cols):
    from utils import preprocess, encode_for_model
    df_p = preprocess(df_new)
    X, _ = encode_for_model(df_p, target_col=None)

    # align columns
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols]

    preds = model.predict(X)
    labels = le.inverse_transform(preds)
    return labels
