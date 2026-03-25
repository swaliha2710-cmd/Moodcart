import streamlit as st
import pandas as pd
from utils import load_data
from eda import run_eda
from models import train_classification, train_regression, train_clustering, association_mining, save_model, load_model, predict_new

st.set_page_config(page_title="MoodCart Analytics", layout="wide")

st.title("MoodCart Analytics Dashboard")

menu = st.sidebar.selectbox("Select Module", [
    "Upload Data",
    "EDA",
    "Classification",
    "Regression",
    "Clustering",
    "Association Rules",
    "Predict New Customers"
])

if "data" not in st.session_state:
    st.session_state["data"] = None

if menu == "Upload Data":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = load_data(file)
        st.session_state["data"] = df
        st.success("Data loaded successfully!")
        st.write(df.head())

elif menu == "EDA":
    if st.session_state["data"] is not None:
        run_eda(st.session_state["data"])
    else:
        st.warning("Upload data first.")

elif menu == "Classification":
    if st.session_state["data"] is not None:
        results, best_model, le, feats = train_classification(st.session_state["data"])
        st.write(pd.DataFrame(results))
        save_model(best_model, le, feats)
        st.success("Best model saved!")
    else:
        st.warning("Upload data first.")

elif menu == "Regression":
    if st.session_state["data"] is not None:
        scores, best_model = train_regression(st.session_state["data"])
        st.write(scores)
    else:
        st.warning("Upload data first.")

elif menu == "Clustering":
    if st.session_state["data"] is not None:
        labels, _ = train_clustering(st.session_state["data"])
        st.write("Cluster counts:", pd.Series(labels).value_counts())
    else:
        st.warning("Upload data first.")

elif menu == "Association Rules":
    if st.session_state["data"] is not None:
        rules = association_mining(st.session_state["data"])
        st.write(rules.head(20))
    else:
        st.warning("Upload data first.")

elif menu == "Predict New Customers":
    st.write("Upload new customer data")
    file = st.file_uploader("Upload CSV for prediction", type=["csv"], key="pred")
    if file:
        df_new = pd.read_csv(file)
        model, le, feats = load_model()
        preds = predict_new(df_new, model, le, feats)
        df_new["Predicted_Interest"] = preds
        st.write(df_new.head())
