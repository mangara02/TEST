import streamlit as st
import numpy as np
import pandas as pd
import joblib

label_encoder_drugs = joblib.load('led.pkl')
label_encoder_branch = joblib.load('leb.pkl')
model = joblib.load('rfr.pkl')

def main():
    st.title("Drug Quantity Prediction App")
    st.write("This app predicts the quantity of drugs to be ordered based on historical data.")

    drug_brands_options = label_encoder_drugs.classes_
    selected_drug_brand = st.selectbox("Select Drug Brand", drug_brands_options)

    branch_options = label_encoder_branch.classes_
    selected_branch = st.selectbox("Select Branch", branch_options)

    selected_month = st.slider("Select a month", 1, 12, 1)
    Adjusted_Qty = st.number_input("Enter a float value:", step=0.1)
    monthly_avg = st.number_input("Enter a float value:", step=0.1)

    input_data = pd.DataFrame([{
        'Drug Brands': [selected_drug_brand],
        'Branch': [selected_branch],
        'Adjusted Qty': [Adjusted_Qty],
        'month': [selected_month],
        'monthly_avg': [monthly_avg]
    }])
    st.dataframe(input_data)

    prediction = model.predict(input_data)
    rounded_prediction = np.ceil(prediction[0])

    st.write("Predicted Quantity:", rounded_prediction)

if __name__ == '__main__':
    main()
