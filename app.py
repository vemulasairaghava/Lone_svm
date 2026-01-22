import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Smart Loan Approval System",
    layout="centered"
)

# -----------------------------
# Title & Description
# -----------------------------
st.title("🏦 Smart Loan Approval System")
st.write(
    """
    This system uses **Support Vector Machines (SVM)** to predict
    whether a loan should be **approved or rejected** based on
    applicant financial and credit details.
    """
)

# -----------------------------
# Load dataset (same as Colab)
# -----------------------------
df = pd.read_csv("loan_approved_cleaned.csv")

X = df[['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Self_Employed']]
y = df['Loan_Status (Approved)']

# -----------------------------
# Train-test split & scaling
# -----------------------------
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# -----------------------------
# Sidebar – User Inputs
# -----------------------------
st.sidebar.header("📝 Applicant Details")

income = st.sidebar.number_input(
    "Applicant Income",
    min_value=0,
    step=500
)

loan_amount = st.sidebar.number_input(
    "Loan Amount",
    min_value=0,
    step=500
)

credit_history = st.sidebar.radio(
    "Credit History",
    ["Yes", "No"]
)

self_employed = st.sidebar.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

credit_val = 1 if credit_history == "Yes" else 0
self_emp_val = 1 if self_employed == "Yes" else 0

# -----------------------------
# Model Selection
# -----------------------------
st.subheader("⚙️ Select SVM Kernel")

kernel_choice = st.radio(
    "Choose Kernel",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"]
)

if kernel_choice == "Linear SVM":
    model = SVC(kernel='linear', C=1, probability=True)

elif kernel_choice == "Polynomial SVM":
    model = SVC(kernel='poly', degree=3, C=1, probability=True)

else:
    model = SVC(kernel='rbf', gamma='scale', C=1, probability=True)

# Train selected model
model.fit(x_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Check Loan Eligibility"):

    user_data = np.array([[
        income,
        loan_amount,
        credit_val,
        self_emp_val
    ]])

    user_data_scaled = scaler.transform(user_data)

    prediction = model.predict(user_data_scaled)[0]
    confidence = model.predict_proba(user_data_scaled).max()

    st.subheader("📊 Loan Decision")

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"**Kernel Used:** {kernel_choice}")
    st.write(f"**Model Confidence:** {confidence*100:.2f}%")

    # -----------------------------
    # Business Explanation
    # -----------------------------
    st.subheader("📌 Business Explanation")

    if prediction == 1:
        st.write(
            "Based on the applicant’s **income stability and credit history**, "
            "the model predicts a **low risk of default**, making the applicant "
            "eligible for loan approval."
        )
    else:
        st.write(
            "Based on **credit history and financial risk indicators**, "
            "the model predicts a **higher probability of default**, "
            "so approving the loan may not be financially safe."
        )
