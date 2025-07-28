import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Health App Dashboard", layout="centered")

# Title
st.title("🧠 Health App Cluster Dashboard")
st.markdown("Get insights from your health app users based on clustering analysis.")

# Upload File
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Show Data
    st.subheader("📄 Sample Data")
    st.dataframe(df.head())

    # Perform Clustering
    st.subheader("🔄 Clustering Data")
    # Select features for clustering
    features = ['Age', 'Monthly Income', 'App Usage Time (hrs)', 'Health Awareness (%)']
    X = df[features]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply K-means clustering (e.g., 3 clusters)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Cluster Group Summary
    st.subheader("📊 Cluster Averages")
    summary = df.groupby("Cluster").mean().reset_index()
    st.dataframe(summary)

    # Plot 1: Monthly Income by Cluster
    st.subheader("💰 Monthly Income by Cluster")
    fig1, ax1 = plt.subplots()
    sns.barplot(data=summary, x="Cluster", y="Monthly Income", ax=ax1)
    st.pyplot(fig1)

    # Plot 2: App Usage Time
    st.subheader("⏱️ App Usage Time by Cluster")
    fig2, ax2 = plt.subplots()
    sns.barplot(data=summary, x="Cluster", y="App Usage Time (hrs)", ax=ax2)
    st.pyplot(fig2)

    # Plot 3: Health Awareness
    st.subheader("🏥 Health Awareness by Cluster")
    fig3, ax3 = plt.subplots()
    sns.barplot(data=summary, x="Cluster", y="Health Awareness (%)", ax=ax3)
    st.pyplot(fig3)

    # Show Personas
    st.subheader("🧑‍🤝‍🧑 Cluster Personas")
    personas = {
        0: "Cluster 0: Young, high-income users with low app usage and awareness.",
        1: "Cluster 1: Middle-aged users with moderate income and high health awareness.",
        2: "Cluster 2: Young, high-income users with high app usage but low awareness.",
    }
    for i in summary['Cluster']:
        st.markdown(f"**Cluster {i}:** {personas.get(i, 'Description not available')}")

    st.success("Analysis Complete! 🎉")
else:
    st.info("Please upload your CSV file to begin.")