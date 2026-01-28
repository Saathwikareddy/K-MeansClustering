import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #f5f7fa;
    font-family: 'Segoe UI', sans-serif;
}

/* Title styling */
h1 {
    color: #1f7a1f;
    text-align: center;
    font-weight: 700;
}

/* Section headers */
h2, h3 {
    color: #2c3e50;
    margin-top: 30px;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #eaf4ea;
    padding: 20px;
}

/* Buttons */
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
    color: white;
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 10px;
    padding: 10px;
}

/* Info box */
.stAlert {
    border-radius: 10px;
}

/* Plot container */
.css-1kyxreq {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------
# App Title & Description
# ------------------------------
st.title("🟢 Customer Segmentation Dashboard")
st.write(
    "This system uses **K-Means Clustering** to group customers based on their purchasing behavior and similarities.\n\n"
    "👉 Discover hidden customer groups without predefined labels."
)

# ------------------------------
# Load Dataset
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('Wholesale customers data.csv')
    return df

df = load_data()

# Clean column names (replace spaces with underscores)
df.columns = df.columns.str.replace(' ', '_')

# Select only numerical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# ------------------------------
# Sidebar – Input Section
# ------------------------------
st.sidebar.header("🔧 Clustering Controls")

feature1 = st.sidebar.selectbox("Select Feature 1", num_cols)
feature2 = st.sidebar.selectbox("Select Feature 2", num_cols, index=1)

k = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=10, value=3)

random_state = st.sidebar.number_input("Random State (Optional)", value=42)

run = st.sidebar.button("🟦 Run Clustering")

# ------------------------------
# Clustering Logic
# ------------------------------
if run:
    X = df[[feature1, feature2]]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means Model
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    clusters = kmeans.fit_predict(X_scaled)

    df['Cluster'] = clusters
    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    # ------------------------------
    # Visualization Section
    # ------------------------------
    st.subheader("📊 Cluster Visualization")

    fig, ax = plt.subplots()
    scatter = ax.scatter(df[feature1], df[feature2], c=df['Cluster'])
    ax.scatter(centers[:, 0], centers[:, 1], s=200, marker='X')

    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.set_title("Customer Clusters")

    st.pyplot(fig)

    # ------------------------------
    # Cluster Summary Section
    # ------------------------------
    st.subheader("📋 Cluster Summary")

    summary = df.groupby('Cluster')[[feature1, feature2]].mean()
    summary['Customer_Count'] = df['Cluster'].value_counts().sort_index()

    st.dataframe(summary)

    # ------------------------------
    # Business Interpretation Section
    # ------------------------------
    st.subheader("💡 Business Interpretation")

    for cluster_id in summary.index:
        avg1 = summary.loc[cluster_id, feature1]
        avg2 = summary.loc[cluster_id, feature2]

        st.write(
            f"🟢 **Cluster {cluster_id}:** Customers showing similar purchasing behavior with "
            f"average {feature1} = {avg1:.2f} and {feature2} = {avg2:.2f}."
        )

    # ------------------------------
    # User Guidance Section
    # ------------------------------
    st.info(
        "Customers in the same cluster exhibit similar purchasing behaviour and "
        "can be targeted with similar business strategies."
    )

else:
    st.warning("👈 Please select features and click **Run Clustering** to view results.")

