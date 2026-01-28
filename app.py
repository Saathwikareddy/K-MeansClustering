import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ------------------------------
# CSS Styling
# ------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #f5f7fa;
    font-family: 'Segoe UI', sans-serif;
}
h1 {
    color: #1f7a1f;
    text-align: center;
    font-weight: 700;
}
h2, h3 {
    color: #2c3e50;
    margin-top: 30px;
}
section[data-testid="stSidebar"] {
    background-color: #eaf4ea;
    padding: 20px;
}
.stButton > button {
    background-color: #2ecc71;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
    border: none;
}
.stButton > button:hover {
    background-color: #27ae60;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Title & Description
# ------------------------------
st.title("🟢 Customer Segmentation Dashboard")
st.write(
    "This system uses **K-Means Clustering** to group customers based on their "
    "purchasing behavior and similarities.\n\n"
    "👉 Discover hidden customer groups without predefined labels."
)

# ------------------------------
# Load Dataset
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Wholesale customers data.csv")
    return df

df = load_data()

# Clean column names
df.columns = df.columns.str.replace(" ", "_")

# Numerical columns
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

# ------------------------------
# Sidebar Controls
# ------------------------------
st.sidebar.header("🔧 Clustering Controls")

features = st.sidebar.multiselect(
    "Select Features (Minimum 2)",
    num_cols,
    default=num_cols[:2]
)

k = st.sidebar.slider("Number of Clusters (K)", 2, 10, 3)
random_state = st.sidebar.number_input("Random State (Optional)", value=42)
run = st.sidebar.button("🟦 Run Clustering")

# ------------------------------
# Clustering Logic
# ------------------------------
if run:
    if len(features) < 2:
        st.error("❗ Please select at least two features.")
    else:
        # Feature selection
        X = df[features]

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # K-Means
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        df["Cluster"] = kmeans.fit_predict(X_scaled)

        # Cluster centers
        centers = scaler.inverse_transform(kmeans.cluster_centers_)

        # First two features for visualization
        f1, f2 = features[0], features[1]

        # ------------------------------
        # Visualization
        # ------------------------------
        st.subheader("📊 Cluster Visualization")

        fig, ax = plt.subplots()
        ax.scatter(df[f1], df[f2], c=df["Cluster"])
        ax.scatter(
            centers[:, features.index(f1)],
            centers[:, features.index(f2)],
            s=200,
            marker="X"
        )

        ax.set_xlabel(f1)
        ax.set_ylabel(f2)
        ax.set_title("Customer Clusters")
        st.pyplot(fig)

        # ------------------------------
        # Cluster Summary
        # ------------------------------
        st.subheader("📋 Cluster Summary")

        summary = df.groupby("Cluster")[features].mean()
        summary["Customer_Count"] = df["Cluster"].value_counts().sort_index()
        st.dataframe(summary)

        # ------------------------------
        # Business Interpretation
        # ------------------------------
        st.subheader("💡 Business Interpretation")

        for cid in summary.index:
            st.write(
                f"🟢 **Cluster {cid}:** Customers with similar purchasing behaviour "
                f"across the selected features."
            )

        st.info(
            "Customers in the same cluster exhibit similar purchasing behaviour "
            "and can be targeted with similar business strategies."
        )

else:
    st.warning("👈 Please select features and click **Run Clustering** to view results.")
