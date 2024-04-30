import streamlit as st
import pandas as pd
import math
import joblib

label_encoder_drugs = joblib.load('led.pkl')
label_encoder_branch = joblib.load('leb.pkl')
model = joblib.load('xgb.pkl')

def main():
    st.title("Drug Quantity Prediction App")
    st.write("This app predicts the quantity of drugs to be ordered based on historical data.")

    # Drug Brands input
    drug_brands_options = label_encoder_drugs.classes_
    selected_drug_brand = st.selectbox("Select Drug Brand", drug_brands_options)

    # Branch input
    branch_options = label_encoder_branch.classes_
    selected_branch = st.selectbox("Select Branch", branch_options)

    # Month input
    selected_month = st.slider("Select a month", 1, 12, 1)

    # Prediction logic
    input_data = pd.DataFrame({
        'Drug Brands': [selected_drug_brand],
        'Branch': [selected_branch],
        'month': [selected_month]
    })

    input_data_o = input_data.apply(pd.to_numeric, errors='coerce')
    input_matrix = xgb.DMatrix(input_data_o)

    prediction = model.predict(input_matrix)
    rounded_prediction = math.ceil(prediction[0])

    st.write("Predicted Quantity:", rounded_prediction)

if __name__ == '__main__':
    main()
