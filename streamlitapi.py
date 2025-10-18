import streamlit as st
import pickle
import numpy as np

# Minimal version with forced styling
st.markdown("""
<style>
    body { background-color: white !important; color: black !important; }
    .stApp { background-color: white !important; }
</style>
""", unsafe_allow_html=True)

st.title("CUSTOMER CHURN PREDICTION")
st.write("Enter customer details:")

# Simple inputs
Contract = st.number_input('Contract', value=1.0)
Online_Security_No = st.number_input('Online Security No', value=0.0)
Dependents_Yes = st.number_input('Dependents Yes', value=0.0)
Tech_Support_No = st.number_input('Tech Support No', value=0.0)
Internet_Service_Fiber_optic = st.number_input('Fiber Optic', value=0.0)
Payment_Method_Electronic_check = st.number_input('Electronic Check', value=0.0)
Total_Charges = st.number_input('Total Charges', value=2000.0)
Tenure_Months = st.number_input('Tenure Months', value=12.0)
Monthly_Charges = st.number_input('Monthly Charges', value=70.0)
CLTV = st.number_input('CLTV', value=4000.0)

if st.button('PREDICT'):
    try:
        model = pickle.load(open("optimized_churn_model.pkl", 'rb'))
        features = [[Contract, Online_Security_No, Dependents_Yes, Tech_Support_No,
                    Internet_Service_Fiber_optic, Payment_Method_Electronic_check,
                    Total_Charges, Tenure_Months, Monthly_Charges, CLTV]]
        
        prediction = model.predict(features)[0]
        
        if prediction == 1:
            st.error("🚨 CUSTOMER WILL CHURN")
        else:
            st.success("✅ CUSTOMER WILL STAY")
            
    except Exception as e:
        st.error(f"Error: {e}")
