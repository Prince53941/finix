import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FINX Decision Engine | CEO Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. PROFESSIONAL STYLING (CSS) ---
st.markdown("""
    <style>
    /* Main Background */
    .main { background-color: #f4f6f9; }
    
    /* Card Styling */
    .st-emotion-cache-1r6slb0 {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        font-weight: bold;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
    }
    
    /* Success/Error Message Styling */
    .success-box {
        padding: 20px; background-color: #d4edda; color: #155724; 
        border-left: 6px solid #28a745; border-radius: 5px; margin-bottom: 20px;
    }
    .error-box {
        padding: 20px; background-color: #f8d7da; color: #721c24; 
        border-left: 6px solid #dc3545; border-radius: 5px; margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. BACKEND LOGIC (Synthetic Data & Model) ---
@st.cache_data
def load_model():
    np.random.seed(42)
    # Generate Training Data
    num_rows = 3000
    cibil = np.random.randint(300, 900, num_rows)
    income = np.random.randint(20000, 300000, num_rows)
    loan = np.random.randint(50000, 5000000, num_rows)
    term = np.random.choice([12, 24, 36, 48, 60], num_rows)
    
    # Simple Logic for Training the AI
    # (In real life, this logic is hidden, but here we simulate it)
    # Rule: If (Loan/Income) is high OR CIBIL is low -> Default
    
    df = pd.DataFrame({'CIBIL': cibil, 'Income': income, 'Loan': loan, 'Term': term})
    
    # Calculate Risk Factors
    df['EMI_Ratio'] = (df['Loan'] / df['Term']) / df['Income']
    
    # Define Target: 1 (Default) if CIBIL < 650 OR EMI takes > 50% of income
    conditions = [
        (df['CIBIL'] < 650) | (df['EMI_Ratio'] > 0.50),
        (df['CIBIL'] >= 650) & (df['EMI_Ratio'] <= 0.50)
    ]
    df['Status'] = np.select(conditions, [1, 0]) # 1 = Default/Reject, 0 = Approve
    
    # Train Model
    X = df[['CIBIL', 'Income', 'Loan', 'Term']]
    y = df['Status']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = load_model()

# --- 4. SIDEBAR (Inputs) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1534/1534082.png", width=50)
    st.markdown("### **FINX Loan Officer Portal**")
    st.markdown("---")
    
    st.header("👤 Applicant Details")
    cibil = st.slider("Credit Score (CIBIL)", 300, 900, 750)
    income = st.number_input("Monthly Income (₹)", 15000, 1000000, 60000, step=5000)
    loan = st.number_input("Loan Amount Requested (₹)", 50000, 10000000, 500000, step=25000)
    term = st.selectbox("Loan Tenure (Months)", [12, 24, 36, 48, 60, 120, 240])
    
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("GENERATE DECISION ⚡")
    st.markdown("---")
    st.caption("v2.5 | Enterprise Edition")

# --- 5. MAIN DASHBOARD ---
st.title("🏦 Credit Decision Memo")
st.markdown(f"**Date:** {pd.Timestamp.now().strftime('%d %B, %Y')} | **Applicant ID:** #FINX-{np.random.randint(1000,9999)}")
st.divider()

if analyze_btn:
    # A. Calculate Financial Metrics (The Math behind the decision)
    monthly_emi = loan / term # Simple EMI for display
    emi_income_ratio = (monthly_emi / income) * 100 # Percentage of salary going to loan
    
    # B. AI Prediction
    input_data = pd.DataFrame([[cibil, income, loan, term]], columns=['CIBIL', 'Income', 'Loan', 'Term'])
    prediction = model.predict(input_data)[0] # 0 = Approve, 1 = Reject
    probability = model.predict_proba(input_data)[0][1] # Risk Score
    
    # C. Logic Generator (The "WHY")
    reasons = []
    if cibil < 650:
        reasons.append(f"❌ **Credit Score Critical:** Applicant's CIBIL ({cibil}) is below the bank's minimum threshold of 650.")
    elif cibil < 750:
        reasons.append(f"⚠️ **Credit Score Moderate:** CIBIL ({cibil}) is acceptable but indicates past missed payments.")
        
    if emi_income_ratio > 50:
        reasons.append(f"❌ **High Debt Burden:** The EMI (₹{int(monthly_emi):,}) would eat up **{int(emi_income_ratio)}%** of monthly income. (Max allowed is 50%).")
    elif emi_income_ratio > 40:
        reasons.append(f"⚠️ **Tight Budget:** EMI is **{int(emi_income_ratio)}%** of income. This is risky.")
        
    if loan > (income * 20):
        reasons.append(f"❌ **Over-Leveraged:** Loan amount is >20x times the monthly income.")

    # D. DISPLAY DECISION
    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        st.subheader("Official Decision Status")
        
        # 1. REJECTED SCENARIO
        if prediction == 1 or len(reasons) > 0: # If AI says reject OR we found critical logic errors
            st.markdown(f"""
            <div class="error-box">
                <h2>⛔ REJECTED</h2>
                <p>Based on the algorithmic risk assessment, this loan application cannot be processed at this time.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("📝 Reason for Rejection")
            if reasons:
                for r in reasons:
                    st.markdown(r)
            else:
                st.markdown("❌ **Algorithmic Reject:** Pattern matches high-risk historical defaulters.")

        # 2. APPROVED SCENARIO
        else:
            st.markdown(f"""
            <div class="success-box">
                <h2>✅ APPROVED</h2>
                <p>The applicant meets all financial health criteria. Disbursal can be initiated immediately.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("🌟 Assessment Highlights")
            st.markdown(f"✔ **Strong Credit Profile:** CIBIL {cibil} is healthy.")
            st.markdown(f"✔ **Affordable EMI:** Loan requires only **{int(emi_income_ratio)}%** of monthly income.")
            st.markdown("✔ **Safe Limits:** Loan amount is within standard eligibility multipliers.")

        # E. DOWNLOAD REPORT BUTTON
        st.subheader("📂 Export")
        
        # Create the text for the file
        decision_status = 'APPROVED ✅' if prediction == 0 and len(reasons) == 0 else 'REJECTED ❌'
        report_text = f"""
        FINX BANK - CREDIT DECISION MEMO
        --------------------------------
        Date: {pd.Timestamp.now().strftime('%d %B, %Y')}
        Applicant ID: #FINX-{np.random.randint(1000,9999)}
        
        APPLICANT DETAILS:
        - Monthly Income: Rs. {income}
        - CIBIL Score: {cibil}
        - Requested Loan: Rs. {loan}
        - Tenure: {term} months
        
        DECISION: {decision_status}
        
        REASONING:
        {chr(10).join(reasons) if reasons else 'Applicant meets all financial criteria.'}
        
        FINANCIAL ANALYSIS:
        - Calculated EMI: Rs. {int(monthly_emi)}
        - Debt-to-Income Ratio: {int(emi_income_ratio)}%
        
        --------------------------------
        Generated by FINX AI Decision Engine
        """
        
        st.download_button(
            label="📄 Download Official Decision Letter",
            data=report_text,
            file_name="Loan_Decision_Memo.txt",
            mime="text/plain"
        )

    with col_right:
        # VISUALS (Speedometer)
        st.subheader("Risk Analysis Meter")
        
        # Risk Score (0 to 100)
        risk_score = int(probability * 100)
        if len(reasons) > 0: risk_score = max(risk_score, 75) # Force high risk on bad logic
        if prediction == 0 and len(reasons) == 0: risk_score = min(risk_score, 20) # Force low risk on good logic
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score,
            title = {'text': "Probability of Default (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred" if risk_score > 50 else "green"},
                'steps': [
                    {'range': [0, 30], 'color': "#e6f4ea"}, # Safe
                    {'range': [30, 70], 'color': "#fff3e0"}, # Warning
                    {'range': [70, 100], 'color': "#fce8e6"}], # Danger
                'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': risk_score}
            }
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Financial Summary Table
        st.markdown("### 📊 Financial Summary")
        metrics = {
            "Monthly EMI (Approx)": f"₹ {int(monthly_emi):,}",
            "Debt-to-Income Ratio": f"{int(emi_income_ratio)}%",
            "Total Repayment": f"₹ {int(monthly_emi * term):,}"
        }
        st.table(pd.DataFrame(metrics, index=[0]).T.rename(columns={0: 'Values'}))

else:
    # Welcome Screen
    st.info("👋 Welcome to the Decision Engine. Use the sidebar to input applicant data.")
    st.markdown("### How Decision Logic Works:")
    st.markdown("""
    1. **Credit Score Check:** Must be > 650.
    2. **Affordability Check:** EMI should not exceed 50% of monthly income.
    3. **AI Pattern Matching:** Compares against 3,000+ historical loan records.
    """)
