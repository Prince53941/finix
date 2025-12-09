import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# --- 1. PAGE CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="FINX RiskGuard | Enterprise Edition", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a Professional Look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa; 
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #004e92;
        color: white;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SYNTHETIC DATA & MODEL (Backend Logic) ---
@st.cache_data
def generate_and_train():
    """Generates data and trains the model once."""
    np.random.seed(42)
    num_rows = 2000
    
    # Generate random features
    cibil = np.random.randint(300, 900, num_rows)
    income = np.random.randint(25000, 300000, num_rows)
    loan = np.random.randint(50000, 5000000, num_rows)
    term = np.random.choice([12, 24, 36, 48, 60], num_rows)
    
    # Logic: Risk Score calculation
    risk_score = (900 - cibil) * 1.2 + (loan / income) * 15
    risk_score += np.random.normal(0, 40, num_rows)
    
    # Threshold for default
    threshold = np.percentile(risk_score, 82)
    defaults = [1 if x > threshold else 0 for x in risk_score]
    
    df = pd.DataFrame({
        'CIBIL_Score': cibil,
        'Monthly_Income': income,
        'Loan_Amount': loan,
        'Loan_Term_Months': term,
        'Target': defaults
    })
    
    # Train Model
    X = df[['CIBIL_Score', 'Monthly_Income', 'Loan_Amount', 'Loan_Term_Months']]
    y = df['Target']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, df

model, historical_df = generate_and_train()

# --- 3. SIDEBAR: APPLICANT PROFILE ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2704/2704029.png", width=60)
    st.title("FINX RiskGuard")
    st.caption("Enterprise Credit Assessment System")
    st.divider()
    
    st.header("📝 New Application")
    
    input_cibil = st.slider("CIBIL / Credit Score", 300, 900, 750, help="Range: 300 (Poor) to 900 (Excellent)")
    input_income = st.number_input("Monthly Income (INR)", min_value=10000, value=65000, step=5000)
    input_loan = st.number_input("Requested Loan Amount", min_value=10000, value=500000, step=10000)
    input_term = st.select_slider("Loan Tenure (Months)", options=[12, 24, 36, 48, 60, 84, 120], value=36)
    
    st.divider()
    predict_btn = st.button("RUN RISK ANALYSIS ▶")
    
    st.markdown("---")
    st.caption("© 2025 FINX Tech. Internal Use Only.")

# --- 4. MAIN DASHBOARD AREA ---

# Top Header
st.title("Credit Risk Analysis Dashboard")
st.markdown(f"**Current Session:** {pd.Timestamp.now().strftime('%d-%b-%Y %H:%M')}")
st.divider()

if predict_btn:
    # --- CALCULATION ---
    input_data = pd.DataFrame([[input_cibil, input_income, input_loan, input_term]], 
                              columns=['CIBIL_Score', 'Monthly_Income', 'Loan_Amount', 'Loan_Term_Months'])
    
    # Get Probability of Default (Risk %)
    risk_prob = model.predict_proba(input_data)[0][1] 
    risk_percentage = round(risk_prob * 100, 1)
    
    # Decision Logic
    if risk_percentage < 30:
        status = "APPROVED"
        color = "green"
        msg = "Applicant meets all credit criteria."
    elif risk_percentage < 60:
        status = "REVIEW REQUIRED"
        color = "orange"
        msg = "Moderate risk detected. Manual verification recommended."
    else:
        status = "REJECTED"
        color = "red"
        msg = "High probability of default detected."

    # --- RESULTS SECTION ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # GAUGE CHART (SPEEDOMETER)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Default Risk Probability"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "lightyellow"},
                    {'range': [60, 100], 'color': "#ffcccb"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_percentage}}))
        
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        # TEXT RESULT
        st.subheader("Decision:")
        if status == "APPROVED":
            st.success(f"### ✅ {status}")
        elif status == "REVIEW REQUIRED":
            st.warning(f"### ⚠️ {status}")
        else:
            st.error(f"### ❌ {status}")
            
        st.info(f"**AI Assessment:** {msg}")
        
        # Key Ratios
        r_col1, r_col2 = st.columns(2)
        dti = (input_loan / input_term) / input_income * 100 # Approx Debt-to-Income
        r_col1.metric("Debt-to-Income Ratio", f"{dti:.1f}%", delta="< 40% is good" if dti < 40 else "High", delta_color="inverse")
        r_col2.metric("Credit Score", input_cibil, delta="Excellent" if input_cibil > 750 else "Average")

else:
    # DEFAULT LANDING SCREEN (Before button press)
    st.info("👈 Please enter applicant details in the sidebar to generate a report.")
    
    # Show Historical Trends (To make the screen look busy/professional initially)
    st.subheader("Market Trends & Historical Data")
    
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        fig1 = px.histogram(historical_df, x="CIBIL_Score", nbins=40, title="Distribution of Applicant Credit Scores")
        fig1.update_layout(bargap=0.1)
        st.plotly_chart(fig1, use_container_width=True)
        
    with row1_col2:
        # Create a simplified Status column for visualization
        historical_df['Status'] = historical_df['Target'].apply(lambda x: 'Default' if x==1 else 'Paid')
        fig2 = px.pie(historical_df, names='Status', title="Portfolio Health (Default vs Paid)", color_discrete_sequence=['red', 'green'])
        st.plotly_chart(fig2, use_container_width=True)
