import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

def main():
    st.title("Drug Quantity Prediction App")
    st.write("This app predicts the quantity of drugs to be ordered based on historical data.")

    model = joblib.load('xgb.pkl')
  
    # Date input
    date_input = st.date_input("Select a date for prediction", pd.to_datetime('today'))

    # Drug Brands input
    drug_brands_options = label_encoder_drugs.classes_
    selected_drug_brand = st.selectbox("Select Drug Brand", drug_brands_options)

    # Branch input
    branch_options = label_encoder_branch.classes_
    selected_branch = st.selectbox("Select Branch", branch_options)

    # Month input
    selected_month = pd.to_datetime(date_input).month

    # Prediction logic
    input_data = pd.DataFrame({
        'month': [selected_month],
        'Drug Brands': [selected_drug_brand],
        'Branch': [selected_branch]
    })

    input_data_o = input_data.apply(pd.to_numeric, errors='coerce')
    input_matrix = xgb.DMatrix(input_data_o)

    prediction = model.predict(input_matrix)

    st.write("Predicted Quantity:", prediction[0])

if __name__ == '__main__':
    main()
