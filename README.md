# 🛡️ FINX Decision Engine: AI-Powered Credit Risk Analyzer

### 🚀 Overview
**FINX Decision Engine** is an intelligent web application designed for the **FINX ProjectXpo 2025–26**. It solves the critical banking problem of **Credit Risk Assessment** by using Machine Learning to predict loan defaults.

Instead of manual verification, this tool allows Loan Officers to input applicant details and get an instant **Approved/Rejected** decision with a detailed financial explanation.

### 🎯 Key Features
* **AI Prediction Model:** Uses a Random Forest Classifier to assess risk based on CIBIL score, Income, and Loan Amount.
* **Financial Logic:** Calculates Debt-to-Income (DTI) ratio and EMI affordability in real-time.
* **Explainable AI:** Does not just say "Rejected" – it tells you *why* (e.g., "EMI is 60% of income").
* **Professional Dashboard:** Built with Streamlit, featuring Gauge Charts (Speedometers) and a download feature.
* **Export Decision:** Allows users to download an official "Decision Memo" as a text file.

### 🛠️ Tech Stack
* **Language:** Python 3.9+
* **Frontend:** Streamlit
* **Machine Learning:** Scikit-Learn (Random Forest)
* **Visualization:** Plotly & Pandas

### 📸 Screenshots
*(You can upload a screenshot of your dashboard here later)*

### ⚙️ How to Run Locally
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/finx-decision-engine.git](https://github.com/yourusername/finx-decision-engine.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

---
*Created for FINX ProjectXpo 2025-26.*
