import streamlit as st
import numpy as np
import pandas as pd
import joblib

label_encoder_drugs = joblib.load('led.pkl')
label_encoder_branch = joblib.load('leb.pkl')
model = joblib.load('rfr.pkl')

def main():
    st.title("Drug Quantity Prediction App")
    st.write("This app predicts the quantity of drugs to be ordered based on historical data")

    drug_brands_options = label_encoder_drugs.classes_
    selected_drug_brand = st.selectbox("Select Drug Brand", drug_brands_options)

    branch_options = label_encoder_branch.classes_
    selected_branch = st.selectbox("Select Branch", branch_options)

    Adjusted_Qty = st.slider('Select Adjusted Quantity', -50.0, 250.0, -0.5, 0.5, key=1, help='Enter the quantity based on bought minus sold quantities of drug brand per branch for this month')
    
    selected_month = st.slider("Select month", 1, 12, 1, key=2, help='Enter this month')

    if selected_month == 12:
        next_month = 1
    else:
        next_month = selected_month + 1

    monthly_avg = st.slider('Select Monthly Average', 7.5, 25.5, 16.5, 0.5, key=3, help='Enter the average quantity of drug brand sold per branch for this month')

    selected_drug_brand_encoded = label_encoder_drugs.transform([selected_drug_brand])[0]
    selected_branch_encoded = label_encoder_branch.transform([selected_branch])[0]

    input_data = pd.DataFrame({
        'Drug Brands': [selected_drug_brand_encoded],
        'Branch': [selected_branch_encoded],
        'Adjusted Qty': [Adjusted_Qty],
        'month': [next_month],
        'monthly_avg': [monthly_avg]
    })

    prediction = model.predict(input_data)
    rounded_prediction = np.ceil(prediction[0])

    st.write("Predicted Quantity of drugs to be ordered next month:", rounded_prediction)

if __name__ == '__main__':
    main()
