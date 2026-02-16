import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load Saved Objects

scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
model = joblib.load("kmeans_model.pkl")

# Business-Friendly Labels

kmeans_labels_map = {
    0: "High-Value Customers",
    1: "Mid-Value Customers",
    2: "Low-Value / At-Risk"
}

st.set_page_config(page_title="Customer Segmentation Dashboard")

st.title("🛍️ Customer Segmentation System")
st.write("Segment customers using PCA + KMeans model")

st.markdown("---")

# OPTION SELECTOR

option = st.radio("Choose Prediction Type:", 
                  ["Single Customer Prediction", "Bulk CSV Upload"])

# 1️ SINGLE CUSTOMER PREDICTION


if option == "Single Customer Prediction":

    st.subheader("Enter Customer Details")

    age = st.number_input("Age", min_value=0)
    total_products = st.number_input("Total Products", min_value=0)
    total_purchases = st.number_input("Total Purchases", min_value=0)
    income = st.number_input("Income", min_value=0.0)
    recency = st.number_input("Recency (Days since last purchase)", min_value=0)

    if st.button("Predict Segment"):

        input_data = np.array([[age, total_products, total_purchases, income, recency]])

        input_scaled = scaler.transform(input_data)
        input_pca = pca.transform(input_scaled)
        prediction = model.predict(input_pca)

        cluster_number = prediction[0]
        segment_name = kmeans_labels_map.get(cluster_number)

        st.success(f"Customer Segment: {segment_name}")


# 2️ BULK CSV UPLOAD


else:

    st.subheader("Upload CSV File")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        required_columns = ['Age', 'Total_Products', 
                            'Total_Purchases', 'Income', 'Recency']

        # Check required columns
        if all(col in df.columns for col in required_columns):

            st.write("Preview of Uploaded Data:")
            st.dataframe(df.head())

            # Select features
            X = df[required_columns]

            # Scale
            X_scaled = scaler.transform(X)

            # PCA
            X_pca = pca.transform(X_scaled)

            # Predict
            clusters = model.predict(X_pca)

            df["Cluster_Number"] = clusters
            df["Customer_Segment"] = df["Cluster_Number"].map(kmeans_labels_map)

            st.success("Segmentation Completed ✅")

            st.dataframe(df.head())

            # Download option
            csv = df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="Download Segmented CSV",
                data=csv,
                file_name="segmented_customers.csv",
                mime="text/csv",
            )

        else:
            st.error("CSV must contain these columns:")
            st.write(required_columns)
