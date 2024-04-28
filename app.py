import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

hd = pd.read_csv('/content/Hospital_data.csv')
dd = pd.read_csv('/content/Drugs_data.csv')

hd1 = hd[['Date OUT', 'Drug Brands', 'Drug Qty', 'Branch']]
hd1.rename(columns={'Date OUT': 'Date', 'Drug Qty': 'OUT Qty'}, inplace=True)
dd.rename(columns={'Drugs': 'Drug Brands', 'Qty': 'IN Qty'}, inplace=True)

mdf = pd.merge(dd, hd1, on=['Date', 'Drug Brands', 'Branch'], how='outer')
mdf.fillna(0, inplace=True)

mdf['Adjusted Qty'] = mdf['IN Qty'] - mdf['OUT Qty']
mdf['Date'] = pd.to_datetime(mdf['Date'])

data = mdf.sort_values(by='Date')
mmdf = data.groupby([data['Date'].dt.to_period('M'), 'Drug Brands', 'Branch']).agg({'IN Qty': 'sum', 'OUT Qty': 'sum', 'Adjusted Qty': 'sum'}).reset_index()

label_encoder_drugs = LabelEncoder()
label_encoder_branch = LabelEncoder()

mmdf['Drug Brands'] = label_encoder_drugs.fit_transform(mmdf['Drug Brands'])
mmdf['Branch'] = label_encoder_branch.fit_transform(mmdf['Branch'])

def date_features(df):
    df['year'] = df.Date.dt.year
    df['month'] = df.Date.dt.month
    return df

mmdf = date_features(mmdf)

smmdf = mmdf.copy()
smmdf.drop('Date', axis=1, inplace=True)
smmdf['monthly_avg'] = smmdf.groupby(['Drug Brands','Branch','month'])['OUT Qty'].transform('mean')
smmdf = smmdf.dropna()
monthly_avg = smmdf.groupby(['Drug Brands','Branch','month'])['OUT Qty'].mean().reset_index()
smmdf['rolling_mean'] = smmdf['OUT Qty'].rolling(window=90, min_periods=1).mean()

for df in [smmdf]:
    df.drop(['IN Qty',
             'Adjusted Qty',
             'year',
             'monthly_avg',
             'rolling_mean'],
              axis=1,
              inplace=True)

X_train, X_test, y_train, y_test = train_test_split(smmdf.drop('OUT Qty', axis=1), smmdf['OUT Qty'], random_state=123, test_size=0.2)

X_traino = X_train.apply(pd.to_numeric, errors='coerce')
X_testo = X_test.apply(pd.to_numeric, errors='coerce')
y_traino = y_train.apply(pd.to_numeric, errors='coerce')
y_testo = y_test.apply(pd.to_numeric, errors='coerce')

matrix_train = xgb.DMatrix(X_traino, label=y_train)
matrix_test = xgb.DMatrix(X_testo, label=y_test)

params = {'objective': 'reg:linear', 'eval_metric': 'mae'}
model = xgb.train(params=params, dtrain=matrix_train, num_boost_round=500, early_stopping_rounds=20, evals=[(matrix_test, 'test')])

joblib.dump(model, 'xgboost_model.pkl')

def main():
    st.title("Drug Quantity Prediction App")
    st.write("This app predicts the quantity of drugs to be ordered based on historical data.")

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
