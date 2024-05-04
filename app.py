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

    Adjusted_Qty = st.slider('Fly Ash', 0.0, 200.0, 55.5, 0.5, key=1)
    
    selected_month = st.slider("Select a month", 1, 12, 1, key=2)

    monthly_avg = st.slider('Fly Ash', 0.0, 200.0, 55.5, 0.5, key=3)

    input_data = pd.DataFrame([{
        'Drug Brands': [selected_drug_brand],
        'Branch': [selected_branch],
        'Adjusted Qty': Adjusted_Qty,
        'month': [selected_month],
        'monthly_avg': monthly_avg
    }])
    st.dataframe(input_data)

    prediction = model.predict(input_data)
    rounded_prediction = np.ceil(prediction[0])

    st.write("Predicted Quantity:", rounded_prediction)

if __name__ == '__main__':
    main()
