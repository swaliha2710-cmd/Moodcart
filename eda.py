import streamlit as st
import pandas as pd
import plotly.express as px

def run_eda(df):
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.write(df.describe(include='all'))

    st.subheader("Target Distribution")
    if "Interest_in_MoodCart" in df.columns:
        fig = px.histogram(df, x="Interest_in_MoodCart")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Income vs Spend")
    if "Income" in df.columns and "Monthly_Spend" in df.columns:
        fig = px.box(df, x="Income", y="Monthly_Spend")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mood Distribution")
    if "Mood" in df.columns:
        fig = px.histogram(df, x="Mood")
        st.plotly_chart(fig, use_container_width=True)
