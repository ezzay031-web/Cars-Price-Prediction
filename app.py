# =========================
# CAR PRICE PREDICTION APP
# =========================

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Title
st.title("🚗 Car Price Prediction App")

# Load Data
df = pd.read_csv("car data.csv")

# =========================
# DATA PREPROCESSING
# =========================

df['Fuel_Type'] = df['Fuel_Type'].map({'Petrol':0, 'Diesel':1, 'CNG':2})
df['Selling_type'] = df['Selling_type'].map({'Dealer':0, 'Individual':1})
df['Transmission'] = df['Transmission'].map({'Manual':0, 'Automatic':1})

df['Car_Age'] = 2025 - df['Year']

df.drop(['Car_Name', 'Year'], axis=1, inplace=True)

# Features & Target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Train Model
model = LinearRegression()
model.fit(X, y)

# =========================
# USER INPUT
# =========================

st.header("Enter Car Details")

present_price = st.number_input("Present Price", min_value=0.0)
driven_kms = st.number_input("Driven KMs", min_value=0)

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

owner = st.selectbox("Owner", [0,1,2,3])
year = st.number_input("Year", min_value=2000, max_value=2025)

# Convert input
fuel = {"Petrol":0, "Diesel":1, "CNG":2}[fuel]
seller = {"Dealer":0, "Individual":1}[seller]
transmission = {"Manual":0, "Automatic":1}[transmission]

car_age = 2025 - year

# Prediction
if st.button("Predict Price"):
    data = [[present_price, driven_kms, fuel, seller, transmission, owner, car_age]]
    prediction = model.predict(data)
    st.success(f"💰 Estimated Price: {prediction[0]:.2f} Lakhs")
