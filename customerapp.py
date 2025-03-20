import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load Dataset
@st.cache_data
def load_data():
    file_path = "https://raw.githubusercontent.com/paul2064/customerapp.py/main/customerdata.csv"  # Update with actual path
    df = pd.read_csv(file_path)
    return df

df = load_data()

st.title("Customer Segmentation App")
st.markdown("This app uses K-Means clustering to segment customers based on selected features.")

# Sidebar options
st.sidebar.header("Select Clustering Features")
selected_features = st.sidebar.multiselect("Choose features for clustering", df.columns[2:])

if selected_features:
    # Data Preprocessing
    X = df[selected_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Number of Clusters
    n_clusters = st.sidebar.slider("Select Number of Clusters", 2, 10, 3)
    
    # K-Means Model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Cluster Centers
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Visualization
    st.subheader("Cluster Visualization")
    if len(selected_features) == 2:
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[selected_features[0]], y=df[selected_features[1]], hue=df['Cluster'], palette='viridis')
        ax.set_xlabel(selected_features[0])
        ax.set_ylabel(selected_features[1])
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=300, c='red', marker='X', label='Centroids')
        ax.legend()
        st.pyplot(fig)
    else:
        st.write("Select exactly two features for a scatter plot visualization.")
    
    st.subheader("Clustered Data")
    cluster_filter = st.multiselect("Filter by Cluster", df['Cluster'].unique())
    if cluster_filter:
        filtered_df = df[df['Cluster'].isin(cluster_filter)]
    else:
        filtered_df = df
    st.write(filtered_df)
    
    # Predict New Customer Cluster
    st.sidebar.header("Predict New Customer Cluster")
    new_data = []
    for feature in selected_features:
        new_data.append(st.sidebar.number_input(f"Enter {feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean())))
    
    if st.sidebar.button("Predict Cluster"):
        new_data_scaled = scaler.transform([new_data])
        predicted_cluster = kmeans.predict(new_data_scaled)[0]
        st.sidebar.success(f"Predicted Cluster: {predicted_cluster}")
    
else:
    st.warning("Please select features for clustering.")
