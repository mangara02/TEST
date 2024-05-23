import streamlit as st
import os
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import accelerate
import bitsandbytes as bnb
from pandasai.llm import BambooLLM
from pandasai import SmartDataframe
import joblib

label_encoder_drugs = joblib.load('led.pkl')
label_encoder_branch = joblib.load('leb.pkl')
model = joblib.load('rfr.pkl')

def main():

    st.set_page_config(
        page_title="Quantity Analysis",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    page = st.sidebar.radio("**Go to:**", ("Introduction :rocket:", "Descriptive analytics :bar_chart:", "Predictive analytics :chart_with_upwards_trend:"))

    if page == "Introduction :rocket:":
        st.title("COMING SOON")

    elif page == "Descriptive analytics :bar_chart:":
        hd = pd.read_csv('Hospital_data.csv')
        dd = pd.read_csv('Drugs_data.csv')

        hd1 = hd[['Date OUT', 'Drug Brands', 'Drug Qty', 'Branch']]
        hd1.rename(columns={'Date OUT': 'Date', 'Drug Qty': 'OUT Qty'}, inplace=True)
        dd.rename(columns={'Drugs': 'Drug Brands', 'Qty': 'IN Qty'}, inplace=True)
        
        mdf = pd.merge(dd, hd1, on=['Date', 'Drug Brands', 'Branch'], how='outer')
        
        mdf.fillna(0, inplace=True)
        
        mdf['Buy Quantity - Sell Quantity'] = mdf['IN Qty'] - mdf['OUT Qty']
        
        mdf['Date'] = pd.to_datetime(mdf['Date'])
        data = mdf.sort_values(by='Date')
        data.rename(columns={'OUT Qty': 'Sell Quantity', 'IN Qty': 'Buy Quantity'}, inplace=True)

        mmdf = data.groupby([data['Date'].dt.to_period('M'), 'Drug Brands', 'Branch']).agg({'Buy Quantity': 'sum', 'Sell Quantity': 'sum', 'Buy Quantity - Sell Quantity': 'sum'}).reset_index()

        mmdf['year'] = mmdf.Date.dt.year
        mmdf['month'] = mmdf.Date.dt.month
        
        smmdf = mmdf.copy()
        smmdf.drop('Date', axis=1, inplace=True)
        
        smmdf['monthly_avg'] = smmdf.groupby(['Drug Brands','Branch','month'])['Sell Quantity'].transform('mean')
        
        monthly_avg = smmdf.groupby(['Drug Brands','Branch','month'])['Sell Quantity'].mean().reset_index()

        distinct_drug_brands = smmdf['Drug Brands'].unique()
        distinct_branches = smmdf['Branch'].unique()

        def get_quantity_info(month, year, drug_brand, branch):
            try:
                info = smmdf[(smmdf['month'] == month) &
                          (smmdf['year'] == year) &
                          (smmdf['Drug Brands'] == drug_brand) &
                          (smmdf['Branch'] == branch)]
                if len(info) > 0:
                    return f"Quantity of {drug_brand} on month {month} year {year} at {branch} is: {info['Sell Quantity'].iloc[0]}"
                else:
                    return "Data not found for the given inputs."
            except Exception as e:
                return f"Error: {str(e)}"

        bamboo_llm = BambooLLM()
        sdf = SmartDataframe(smmdf, config={"llm": bamboo_llm})

        def chat_with_smart_dataframe(input_text):
            return sdf.chat(input_text)
        
        st.title("Drug Quantity Information")

        use_llm = st.checkbox('Enable LLM Functionality')

        if use_llm:
            api_key = st.text_input("Enter your OpenAI API key:", type="password")
    
            if st.button("Save API Key"):
                if api_key:
                    os.environ["PANDASAI_API_KEY"] = api_key
                    input_text = st.text_input("Ask a question about the data:")
                    answer = chat_with_smart_dataframe(input_text)
                    st.write(answer)
                else:
                    st.error("Please enter your OpenAI API key.")
                
        else:
            st.write("Select parameters to get Sell Quantity")
            
            month_input = st.slider("Month", 1, 12, 1, key=1)
            year_input = st.slider("year", 2020, 2024, 2020, key=2)
            drug_brand_input = st.selectbox("Drug Brand", options=distinct_drug_brands)
            branch_input = st.selectbox("Branch", options=distinct_branches)
            
            if st.button("Get Quantity Info"):
                result = get_quantity_info(month_input, year_input, drug_brand_input, branch_input)
                st.write(result)

    elif page == "Predictive analytics :chart_with_upwards_trend:":
        st.title("Drug Quantity Prediction App")
        st.write("This app predicts the quantity of drugs to be ordered based on historical data")
    
        drug_brands_options = label_encoder_drugs.classes_
        selected_drug_brand = st.selectbox("Select Drug Brand", drug_brands_options)
    
        branch_options = label_encoder_branch.classes_
        selected_branch = st.selectbox("Select Branch", branch_options)
    
        Adjusted_Qty = st.slider('Select Adjusted Quantity', -50.0, 250.0, -0.5, 0.5, key=1, help='Enter the quantity based on bought minus sold quantities of selected drug brand per selected branch for this month')
        
        selected_month = st.slider("Select month", 1, 12, 1, key=2, help='Enter this month')
    
        if selected_month == 12:
            next_month = 1
        else:
            next_month = selected_month + 1
    
        monthly_avg = st.slider('Select Monthly Average', 7.5, 25.5, 16.5, 0.5, key=3, help='Enter the average quantity of selected drug brand sold per selected branch for this month')
    
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

        if st.button("Predict"):
            st.write("Predicted Quantity of drugs to be ordered next month:", rounded_prediction)

if __name__ == '__main__':
    main()
