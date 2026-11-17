import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Segmentasi Pelanggan", layout="centered")

@st.cache_resource
def load_package():
    path = "deployment_model/clustering_deployment.joblib"
    if not os.path.exists(path):
        path = "clustering_deployment.joblib"
    return joblib.load(path)

package = load_package()
model = package["model"]
scaler = package["scaler"]
feature_columns = package["feature_columns"]

st.title("Segmentasi Pelanggan Online Retail")
st.write("Prediksi cluster pelanggan menggunakan K-Means (RFM)")

recency = st.number_input("Recency (hari)", min_value=0, value=10)
frequency = st.number_input("Frequency (transaksi)", min_value=1, value=12)
monetary = st.number_input("Monetary (£)", min_value=1.0, value=3500.0)

if st.button("Prediksi Cluster"):
    data = pd.DataFrame({
        "Recency": [recency],
        "Frequency_log": [np.log1p(frequency)],
        "Monetary_log": [np.log1p(monetary)]
    })
    data_scaled = scaler.transform(data[feature_columns])
    cluster = model.predict(data_scaled)[0]
    st.success(f"Pelanggan masuk ke Cluster {cluster}")
