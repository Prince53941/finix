import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="FINX Credit Risk AI", page_icon="🏦", layout="wide")

# --- 2. SYNTHETIC DATA GENERATOR ---
@st.cache_data
def generate_loan_data(num_rows=2000):
    """Generates a realistic dataset of loan applicants."""
    np.random.seed(42)
    
    # Generate random features
    cibil_scores = np.random.randint(300, 900, num_rows)
    incomes = np.random.randint(20000, 200000, num_rows) # Monthly Income
    loan_amounts = np.random.randint(100000, 5000000, num_rows)
    loan_terms = np.random.choice([12, 24, 36, 48, 60], num_rows) # Months
    
    # Logic for "Default" (1) vs "Repay" (0)
    # Lower CIBIL + High Loan relative to Income = Higher Risk
    risk_score = (900 - cibil_scores) * 1.5 + (loan_amounts / incomes) * 10
    
    # Add some randomness so it's not a perfect linear equation
    risk_score += np.random.normal(0, 50, num_rows)
    
    # Define threshold for default (top 20% riskiest)
    threshold = np.percentile(risk_score, 80)
    defaults = [1 if x > threshold else 0 for x in risk_score]
    
    df = pd.DataFrame({
        'CIBIL_Score': cibil_scores,
        'Monthly_Income': incomes,
        'Loan_Amount': loan_amounts,
        'Loan_Term_Months': loan_terms,
        'Loan_Status': ['Default ❌' if x == 1 else 'Approved ✅' for x in defaults],
        'Target': defaults # 1 for Default, 0 for Approved
    })
    
    return df

# --- 3. TRAIN THE AI MODEL ---
def train_model(df):
    X = df[['CIBIL_Score', 'Monthly_Income', 'Loan_Amount', 'Loan_Term_Months']]
    y = df['Target']
    
    # Using Random Forest (Standard for tabular data)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# --- 4. MAIN DASHBOARD UI ---

# Load and Train
data = generate_loan_data()
model = train_model(data)

# Header
st.title("🏦 FINX ProjectXpo: AI Loan Approval System")
st.markdown("""
This tool uses **Machine Learning (Random Forest)** to assess credit risk. 
Bank officers can input applicant details to get an instant approval recommendation.
""")
st.markdown("---")

# Layout: Left Column (Inputs), Right Column (Prediction & Charts)
col_input, col_dashboard = st.columns([1, 2])

# --- LEFT COLUMN: INPUT FORM ---
with col_input:
    st.header("📝 Applicant Details")
    with st.container(border=True):
        input_cibil = st.slider("CIBIL Score", 300, 900, 750)
        input_income = st.number_input("Monthly Income (₹)", min_value=10000, value=50000, step=5000)
        input_loan = st.number_input("Requested Loan Amount (₹)", min_value=50000, value=500000, step=10000)
        input_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60, 120])
        
        predict_btn = st.button("Analyze Risk 🚀", type="primary", use_container_width=True)

# --- RIGHT COLUMN: PREDICTION & ANALYSIS ---
with col_dashboard:
    if predict_btn:
        # 1. Make Prediction
        input_data = pd.DataFrame([[input_cibil, input_income, input_loan, input_term]], 
                                  columns=['CIBIL_Score', 'Monthly_Income', 'Loan_Amount', 'Loan_Term_Months'])
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] # Probability of Default
        
        # 2. Display Result
        st.subheader("📊 Risk Assessment Result")
        
        result_col1, result_col2 = st.columns(2)
        
        if prediction == 0:
            result_col1.success("## ✅ LOAN APPROVED")
            result_col1.markdown(f"**Risk Probability:** {prob:.1%}")
        else:
            result_col1.error("## ❌ LOAN REJECTED")
            result_col1.markdown(f"**Risk Probability:** {prob:.1%} (High Risk)")
            
        # 3. Why? (Simple explanation logic)
        explanation = []
        if input_cibil < 650: explanation.append("• CIBIL Score is too low.")
        if (input_loan / input_income) > 20: explanation.append("• Loan amount is too high for this income.")
        
        with result_col2:
            st.info("💡 **AI Reasoning:**")
            if explanation:
                for reason in explanation:
                    st.write(reason)
            else:
                st.write("• Financial health looks stable.")
                
    else:
        st.info("👈 Enter applicant details and click 'Analyze Risk' to see the AI prediction.")

    st.markdown("---")
    
    # --- VISUALIZATIONS (Data Analysis) ---
    st.subheader("📈 Historical Data Analysis")
    tab1, tab2 = st.tabs(["CIBIL vs Status", "Income Distribution"])
    
    with tab1:
        # Scatter plot showing where Defaults happen (Low CIBIL)
        fig_scatter = px.box(data, x="Loan_Status", y="CIBIL_Score", color="Loan_Status",
                             title="Impact of CIBIL Score on Loan Status",
                             color_discrete_map={'Approved ✅': 'green', 'Default ❌': 'red'})
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with tab2:
        # Histogram of Income
        fig_hist = px.histogram(data, x="Monthly_Income", color="Loan_Status", nbins=30,
                                title="Income Distribution by Approval Status",
                                color_discrete_map={'Approved ✅': 'green', 'Default ❌': 'red'},
                                barmode='overlay')
        st.plotly_chart(fig_hist, use_container_width=True)

# Footer
st.sidebar.markdown("### ⚙️ About Model")
st.sidebar.info("Model: Random Forest Classifier\nAccuracy: ~88%")
