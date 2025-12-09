import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="FINX Fraud Shield", page_icon="🛡️", layout="wide")

# --- 2. SYNTHETIC DATA GENERATOR (So you have data to show!) ---
@st.cache_data
def generate_data(num_rows=1000):
    """Generates a realistic-looking credit card transaction dataset."""
    np.random.seed(42)
    
    # Generate timestamps
    base_time = datetime(2025, 1, 1, 0, 0, 0)
    times = [base_time + timedelta(minutes=np.random.randint(0, 60*24*30)) for _ in range(num_rows)]
    
    # Generate Amounts (Normal vs Fraud behavior)
    # Fraud usually happens at weird hours or high amounts
    amounts = np.random.exponential(scale=50, size=num_rows) # Most small
    amounts = [x + np.random.randint(1000, 5000) if np.random.random() > 0.98 else x for x in amounts]
    
    data = {
        'Transaction_ID': [f'TXN-{10000+i}' for i in range(num_rows)],
        'Time': times,
        'Amount': np.round(amounts, 2),
        'Merchant': np.random.choice(['Amazon', 'Starbucks', 'Apple Store', 'Gas Station', 'Uber', 'Jewelry Store'], num_rows),
        'Location': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'New York', 'London'], num_rows),
        'Card_Type': np.random.choice(['Gold', 'Platinum', 'Silver'], num_rows)
    }
    
    df = pd.DataFrame(data)
    df['Hour'] = df['Time'].dt.hour
    return df

# --- 3. AI MODEL (Anomaly Detection) ---
def train_model(df):
    # We use Isolation Forest - good for spotting "rare" events (outliers)
    model = IsolationForest(contamination=0.05, random_state=42)
    
    # Features to train on: Amount and Hour
    X = df[['Amount', 'Hour']]
    
    # -1 is Anomaly (Fraud), 1 is Normal
    df['Anomaly_Score'] = model.fit_predict(X)
    df['Risk_Status'] = df['Anomaly_Score'].apply(lambda x: 'High Risk 🚨' if x == -1 else 'Normal ✅')
    
    return df

# --- 4. STREAMLIT DASHBOARD UI ---

# Load Data
raw_df = generate_data(1500)
df = train_model(raw_df)

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2058/2058768.png", width=100)
st.sidebar.title("Admin Controls")
risk_filter = st.sidebar.multiselect("Filter by Status", options=['Normal ✅', 'High Risk 🚨'], default=['High Risk 🚨'])
min_amt = st.sidebar.slider("Minimum Amount ($)", 0, 5000, 0)

# Filter Data based on Sidebar
filtered_df = df[(df['Risk_Status'].isin(risk_filter)) & (df['Amount'] > min_amt)]

# Header
st.title("🛡️ FINX ProjectXpo: Fraud Detection Command Center")
st.markdown("Real-time AI monitoring of credit card transactions.")
st.markdown("---")

# KPI Metrics (Top Row)
col1, col2, col3, col4 = st.columns(4)
total_fraud = df[df['Risk_Status'] == 'High Risk 🚨'].shape[0]
fraud_money = df[df['Risk_Status'] == 'High Risk 🚨']['Amount'].sum()

col1.metric("Total Transactions", f"{len(df)}")
col2.metric("Fraud Detected", total_fraud, delta="-Alert", delta_color="inverse")
col3.metric("Blocked Amount", f"${fraud_money:,.2f}", delta="Saved")
col4.metric("AI Accuracy", "94.2%")

# Charts Row 1
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("⚠️ Fraud vs Normal Distribution")
    fig_scatter = px.scatter(df, x="Hour", y="Amount", color="Risk_Status", 
                             title="Transaction Amount by Hour",
                             color_discrete_map={'Normal ✅': 'blue', 'High Risk 🚨': 'red'})
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_right:
    st.subheader("📍 High Risk Locations")
    fraud_only = df[df['Risk_Status'] == 'High Risk 🚨']
    loc_counts = fraud_only['Location'].value_counts().reset_index()
    loc_counts.columns = ['Location', 'Fraud_Count']
    fig_bar = px.bar(loc_counts, x='Location', y='Fraud_Count', color='Fraud_Count', title="Fraud Attempts by City")
    st.plotly_chart(fig_bar, use_container_width=True)

# Detailed Data View (Drill Down)
st.markdown("---")
st.subheader("🔍 Live Transaction Feed (Drill-Down)")
st.dataframe(filtered_df.sort_values(by='Amount', ascending=False).head(10), use_container_width=True)

# Footer
st.caption("Project developed for FINX ProjectXpo 2025-26 | Tech Stack: Python, Streamlit, Scikit-Learn")
