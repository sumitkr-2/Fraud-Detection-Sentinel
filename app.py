import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Fraud Detection Sentinel",
    page_icon="🛡️",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .metric-card {
        background-color: #262730;
        border-radius: 10px;
        padding: 20px;
        color: white;
    }
    .high-risk {
        color: #FF4B4B;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. Data Loading & Processing ---
@st.cache_data
def generate_sample_data():
    """Generates synthetic data for demonstration if no file is uploaded."""
    np.random.seed(42)
    data_size = 500
    
    states = ['Andhra Pradesh', 'Telangana', 'Karnataka', 'Tamil Nadu']
    districts = [f'District_{i}' for i in range(1, 20)]
    
    data = pd.DataFrame({
        'state': np.random.choice(states, data_size),
        'district': np.random.choice(districts, data_size),
        'pincode': np.random.randint(500001, 599999, data_size),
        'age_18_greater': np.random.randint(50, 1000, data_size),
        'demo_age_17_': np.random.randint(10, 500, data_size),
        'total_bio_updates': np.random.randint(5, 300, data_size) # Added Biometric
    })
    
    # Inject synthetic anomalies (High Biometric + Demographic Spikes)
    anomalies = pd.DataFrame({
        'state': ['Andhra Pradesh'] * 10,
        'district': ['District_X'] * 10,
        'pincode': np.random.randint(500001, 599999, 10),
        'age_18_greater': np.random.randint(2000, 5000, 10),  
        'demo_age_17_': np.random.randint(2000, 4000, 10),
        'total_bio_updates': np.random.randint(1500, 3000, 10) # High Biometric updates
    })
    
    return pd.concat([data, anomalies], ignore_index=True)

def process_uploaded_data(enrolment_file, demographic_file, biometric_file):
    """Processes real uploaded CSV files including Biometric data."""
    try:
        enrolment = pd.read_csv(enrolment_file)
        demographic = pd.read_csv(demographic_file)
        biometric = pd.read_csv(biometric_file)
        
        # 1. Enrolment Aggregation (Adults)
        enr_agg = enrolment.groupby(["state", "district", "pincode"], as_index=False)["age_18_greater"].sum()
        
        # 2. Demographic Aggregation (Adult Updates)
        demo_agg = demographic.groupby(["state", "district", "pincode"], as_index=False)["demo_age_17_"].sum()
        
        # 3. Biometric Aggregation (Total Updates)
        # Calculate total if not already aggregated
        biometric["total_bio_updates"] = biometric["bio_age_5_17"] + biometric["bio_age_17_"]
        bio_agg = biometric.groupby(["state", "district", "pincode"], as_index=False)["total_bio_updates"].sum()
        
        # 4. Merge All Features
        features = pd.merge(enr_agg, demo_agg, on=["state", "district", "pincode"], how="outer")
        features = pd.merge(features, bio_agg, on=["state", "district", "pincode"], how="outer").fillna(0)
        
        return features
    except Exception as e:
        st.error(f"Error processing files: {e}")
        return None

# --- 2. Sidebar: Dataset Selection ---
st.sidebar.header("📂 Data Source")
data_source = st.sidebar.radio("Choose Dataset:", ["Upload My Data", "Use Sample Data (Demo)"])

df_final = None

if data_source == "Upload My Data":
    st.sidebar.info("Upload your CSV files to detect anomalies.")
    enrol_file = st.sidebar.file_uploader("Upload Enrolment CSV", type=["csv"])
    demo_file = st.sidebar.file_uploader("Upload Demographic CSV", type=["csv"])
    bio_file = st.sidebar.file_uploader("Upload Biometric CSV", type=["csv"])
    
    if enrol_file and demo_file and bio_file:
        with st.spinner("Processing Data..."):
            df_final = process_uploaded_data(enrol_file, demo_file, bio_file)
    elif enrol_file or demo_file or bio_file:
        st.sidebar.warning("Please upload ALL 3 files (Enrolment, Demographic, Biometric).")

else:
    df_final = generate_sample_data()
    st.sidebar.success("✅ Loaded Sample Data")

# --- 3. Main Application Logic ---
if df_final is not None:
    
    # --- Anomaly Detection Model ---
    # Now utilizing 3 features!
    model_features = ['age_18_greater', 'demo_age_17_', 'total_bio_updates']
    
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    df_final['anomaly_score'] = model.fit_predict(df_final[model_features])
    
    # -1 is anomaly, 1 is normal
    df_final['risk_label'] = df_final['anomaly_score'].apply(lambda x: 'High Risk 🚨' if x == -1 else 'Normal ✅')
    
    high_risk_df = df_final[df_final['risk_label'] == 'High Risk 🚨']
    
    # --- Dashboard Header ---
    st.title("🛡️ Fraud Detection & Anomaly Dashboard")
    st.markdown("### Real-time analysis of Enrolment, Demographic & Biometric patterns")
    
    # --- Top Level Metrics ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Pincodes Scanned", len(df_final))
    c2.metric("High Risk Locations Detected", len(high_risk_df), delta="-Alert", delta_color="inverse")
    c3.metric("Avg. Fraud Score", f"{len(high_risk_df)/len(df_final)*100:.2f}%")
    
    st.divider()

    # --- 4. Visualizations ---
    col_chart1, col_chart2 = st.columns([2, 1])
    
    with col_chart1:
        st.subheader("🔍 3D Pattern Analysis")
        st.caption("Axes: Enrolments (X), Demographic Updates (Y), Biometric Updates (Z)")
        
        # 3D Scatter Plot for 3 Features
        fig_3d = px.scatter_3d(
            df_final, 
            x="age_18_greater", 
            y="demo_age_17_", 
            z="total_bio_updates",
            color="risk_label",
            hover_data=["state", "district", "pincode"],
            color_discrete_map={'High Risk 🚨': 'red', 'Normal ✅': 'blue'},
            title="Multidimensional Anomaly Clusters",
            height=600
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with col_chart2:
        st.subheader("📍 High Risk Districts")
        # Bar chart of top districts with anomalies
        risk_by_district = high_risk_df['district'].value_counts().head(10).reset_index()
        risk_by_district.columns = ['District', 'Count']
        
        fig_bar = px.bar(
            risk_by_district, 
            x='District', 
            y='Count',
            color='Count',
            color_continuous_scale='Reds',
            title="Top Risky Districts"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- 5. Alerts & Explanations ---
    st.subheader("🚨 Live Risk Alerts")
    
    if not high_risk_df.empty:
        # Create dynamic descriptions for the alerts
        display_df = high_risk_df.copy()
        
        def generate_description(row):
            reasons = []
            if row['age_18_greater'] > 2000:
                reasons.append("High Adult Enrolment")
            if row['demo_age_17_'] > 2000:
                reasons.append("High Demographic Updates")
            if row['total_bio_updates'] > 1500:
                reasons.append("Suspicious Biometric Activity")
            
            if not reasons:
                return "Anomaly detected based on density patterns."
            return "CRITICAL: " + " + ".join(reasons)

        display_df['Risk Description'] = display_df.apply(generate_description, axis=1)
        
        # Show interactive dataframe
        st.dataframe(
            display_df[['state', 'district', 'pincode', 'age_18_greater', 'demo_age_17_', 'total_bio_updates', 'Risk Description']],
            column_config={
                "age_18_greater": st.column_config.NumberColumn("Adult Enrolments"),
                "demo_age_17_": st.column_config.NumberColumn("Demographic Updates"),
                "total_bio_updates": st.column_config.NumberColumn("Biometric Updates"),
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("No high-risk anomalies detected in the current dataset.")

else:
    st.info("👈 Please select a Data Source from the sidebar to begin.")