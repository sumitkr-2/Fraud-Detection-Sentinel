import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.ensemble import IsolationForest
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Fraud Detection Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1E1E1E;
        border-radius: 5px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. Data Loading & Processing ---
@st.cache_data
def generate_sample_data():
    """Generates synthetic data for demonstration."""
    np.random.seed(42)
    data_size = 1000
    
    states = ['Andhra Pradesh', 'Telangana', 'Karnataka', 'Tamil Nadu', 'Maharashtra']
    districts = [f'District_{i}' for i in range(1, 25)]
    
    data = pd.DataFrame({
        'state': np.random.choice(states, data_size),
        'district': np.random.choice(districts, data_size),
        'pincode': np.random.randint(500001, 599999, data_size),
        'age_18_greater': np.random.randint(50, 800, data_size),
        'demo_age_17_': np.random.randint(10, 300, data_size),
        'total_bio_updates': np.random.randint(5, 200, data_size)
    })
    
    # Inject synthetic anomalies
    anomalies = pd.DataFrame({
        'state': ['Andhra Pradesh'] * 15,
        'district': ['District_X_Files'] * 15,
        'pincode': np.random.randint(500001, 599999, 15),
        'age_18_greater': np.random.randint(2500, 5000, 15),  
        'demo_age_17_': np.random.randint(2000, 4500, 15),
        'total_bio_updates': np.random.randint(1000, 3000, 15)
    })
    
    return pd.concat([data, anomalies], ignore_index=True)

def process_uploaded_data(enrolment_file, demographic_file, biometric_file):
    try:
        enrolment = pd.read_csv(enrolment_file)
        demographic = pd.read_csv(demographic_file)
        biometric = pd.read_csv(biometric_file)
        
        # Aggregations
        enr_agg = enrolment.groupby(["state", "district", "pincode"], as_index=False)["age_18_greater"].sum()
        demo_agg = demographic.groupby(["state", "district", "pincode"], as_index=False)["demo_age_17_"].sum()
        
        if "total_bio_updates" not in biometric.columns:
            # Fallback if specific column names differ, tries to sum numeric cols
            numeric_cols = biometric.select_dtypes(include=np.number).columns.tolist()
            # Exclude pin/state codes if they are numeric
            cols_to_sum = [c for c in numeric_cols if c not in ['pincode', 'state_code', 'district_code']]
            biometric["total_bio_updates"] = biometric[cols_to_sum].sum(axis=1)
            
        bio_agg = biometric.groupby(["state", "district", "pincode"], as_index=False)["total_bio_updates"].sum()
        
        features = pd.merge(enr_agg, demo_agg, on=["state", "district", "pincode"], how="outer")
        features = pd.merge(features, bio_agg, on=["state", "district", "pincode"], how="outer").fillna(0)
        
        return features
    except Exception as e:
        st.error(f"Error processing files: {e}")
        return None

# --- 2. Sidebar Setup ---
st.sidebar.title("🛡️ Sentinel Control")
data_source = st.sidebar.radio("Data Source:", ["Upload Data", "Launch Demo Mode"], index=1)

df_final = None

if data_source == "Upload Data":
    st.sidebar.info("Required: 3 CSV files (Enrolment, Demographic, Biometric)")
    enrol_file = st.sidebar.file_uploader("Enrolment CSV", type=["csv"])
    demo_file = st.sidebar.file_uploader("Demographic CSV", type=["csv"])
    bio_file = st.sidebar.file_uploader("Biometric CSV", type=["csv"])
    
    if enrol_file and demo_file and bio_file:
        with st.spinner("Ingesting Data Layers..."):
            df_final = process_uploaded_data(enrol_file, demo_file, bio_file)
else:
    df_final = generate_sample_data()
    st.sidebar.success("✅ Demo Data Loaded")

# --- 3. Main Logic ---
if df_final is not None:
    
    # --- MODEL TRAINING ---
    model_features = ['age_18_greater', 'demo_age_17_', 'total_bio_updates']
    model = IsolationForest(n_estimators=200, contamination=0.04, random_state=42)
    df_final['anomaly_score'] = model.fit_predict(df_final[model_features])
    df_final['risk_label'] = df_final['anomaly_score'].apply(lambda x: 'High Risk' if x == -1 else 'Normal')
    
    # Separate High Risk
    high_risk_df = df_final[df_final['risk_label'] == 'High Risk']

    # Title Area
    c_title, c_logo = st.columns([3,1])
    with c_title:
        st.title("Fraud Detection Sentinel")
        st.caption("AI-Powered Anomaly Detection System for Aadhaar Enrolment Operations")
    
    # --- TABS INTERFACE ---
    tab1, tab2, tab3 = st.tabs(["🚨 Risk Radar (3D)", "📍 District Intel", "🔍 Data Inspector"])

    # === TAB 1: 3D VISUALIZATION & ALERTS ===
    with tab1:
        # Top KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Pincodes", len(df_final))
        kpi2.metric("Flagged Anomalies", len(high_risk_df), delta="Action Required", delta_color="inverse")
        kpi3.metric("Max Biometric Spike", f"{df_final['total_bio_updates'].max()}", "Updates/Month")
        kpi4.metric("Avg Risk Density", f"{(len(high_risk_df)/len(df_final))*100:.1f}%")
        
        st.divider()
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Multidimensional Risk Clusters")
            fig_3d = px.scatter_3d(
                df_final, 
                x="age_18_greater", 
                y="demo_age_17_", 
                z="total_bio_updates",
                color="risk_label",
                hover_data=["state", "district", "pincode"],
                color_discrete_map={'High Risk': '#FF4B4B', 'Normal': '#00CC96'},
                opacity=0.7,
                height=600,
                title="3D Anomaly Detection (Rotate to Explore)"
            )
            st.plotly_chart(fig_3d, use_container_width=True)
            
        with c2:
            st.subheader("Live Anomaly Feed")
            if not high_risk_df.empty:
                # Dynamic Descriptions
                def get_reason(row):
                    reasons = []
                    if row['age_18_greater'] > df_final['age_18_greater'].quantile(0.95): reasons.append("Adult Enrolment Spike")
                    if row['demo_age_17_'] > df_final['demo_age_17_'].quantile(0.95): reasons.append("Excessive Demo Updates")
                    if row['total_bio_updates'] > df_final['total_bio_updates'].quantile(0.95): reasons.append("Bio Update Surge")
                    return ", ".join(reasons) if reasons else "Statistical Outlier"

                display_feed = high_risk_df.copy()
                display_feed['Reason'] = display_feed.apply(get_reason, axis=1)
                
                st.dataframe(
                    display_feed[['pincode', 'district', 'Reason']], 
                    hide_index=True, 
                    use_container_width=True,
                    height=500
                )
            else:
                st.success("System Secure: No anomalies detected.")

    # === TAB 2: DISTRICT INTELLIGENCE ===
    with tab2:
        st.subheader("District-wise Action Plan")
        
        # Calculate Risk Score per District (Ratio of Anomalies to Total Pincodes)
        district_stats = df_final.groupby('district').agg(
            total_pincodes=('pincode', 'count'),
            high_risk_count=('anomaly_score', lambda x: (x == -1).sum())
        ).reset_index()
        
        district_stats['risk_ratio'] = district_stats['high_risk_count'] / district_stats['total_pincodes']
        
        # Assign Action Plans
        def get_action(ratio):
            if ratio > 0.5: return "🔴 Immediate Audit"
            elif ratio > 0.2: return "🟡 Surveillance Watchlist"
            else: return "🟢 Routine Check"
            
        district_stats['Recommended Action'] = district_stats['risk_ratio'].apply(get_action)
        district_stats = district_stats.sort_values('risk_ratio', ascending=False)
        
        col_act1, col_act2 = st.columns([2, 1])
        
        with col_act1:
            fig_bar = px.bar(
                district_stats.head(15), 
                x="risk_ratio", 
                y="district", 
                color="Recommended Action",
                color_discrete_map={
                    "🔴 Immediate Audit": "#FF4B4B",
                    "🟡 Surveillance Watchlist": "#FFA500",
                    "🟢 Routine Check": "#00CC96"
                },
                orientation='h',
                title="Top Districts Requiring Intervention",
                labels={"risk_ratio": "Risk Density Score"}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_act2:
            st.warning("⚠️ Action Required")
            audit_list = district_stats[district_stats['Recommended Action'] == "🔴 Immediate Audit"]
            if not audit_list.empty:
                st.write(f"**{len(audit_list)} Districts** are flagged for immediate physical audit due to >50% anomalous activity.")
                st.dataframe(audit_list[['district', 'high_risk_count']], hide_index=True)
            else:
                st.success("No districts are currently in the Critical Red Zone.")

    # === TAB 3: DATA INSPECTOR ===
    with tab3:
        st.subheader("Deep Dive Analysis")
        
        # Correlation Heatmap
        st.markdown("#### 1. Feature Correlations")
        st.caption("How do different fraud indicators relate to each other?")
        corr = df_final[['age_18_greater', 'demo_age_17_', 'total_bio_updates']].corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Distribution Plots
        st.markdown("#### 2. Data Distributions")
        feature_select = st.selectbox("Select Feature to Inspect:", model_features)
        fig_hist = px.histogram(df_final, x=feature_select, color="risk_label", nbins=50, 
                                title=f"Distribution of {feature_select} (Normal vs Risk)")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Raw Data View
        st.markdown("#### 3. Raw Data Browser")
        st.dataframe(df_final, use_container_width=True)

    # --- Sidebar Download ---
    st.sidebar.divider()
    if not high_risk_df.empty:
        csv = high_risk_df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            "📥 Download Risk Report",
            csv,
            "fraud_risk_report.csv",
            "text/csv",
            key='download-csv'
        )

else:
    st.info("Awaiting Data Stream...")
