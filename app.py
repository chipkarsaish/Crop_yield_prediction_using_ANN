import streamlit as st
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np

## ----------------LOAD MODEL------------------
model = load_model('model.h5')

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="centered"
)

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style="text-align:center;">🌾 Crop Yield Prediction</h1>
    <h4 style="text-align:center;">Using Artificial Neural Network(ANN)<h2>
    <p style="text-align:center; color: gray;">
        Enter agricultural and environmental details to predict crop yield
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- DROPDOWN VALUES ----------------
# (Replace or auto-load these later if needed)

AREA_LIST = [
    'Albania', 'Algeria', 'Angola', 'Argentina', 'Armenia',
    'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain',
    'Bangladesh', 'Belarus', 'Belgium', 'Botswana', 'Brazil',
    'Bulgaria', 'Burkina Faso', 'Burundi', 'Cameroon', 'Canada',
    'Central African Republic', 'Chile', 'Colombia', 'Croatia',
    'Denmark', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador',
    'Eritrea', 'Estonia', 'Finland', 'France', 'Germany', 'Ghana',
    'Greece', 'Guatemala', 'Guinea', 'Guyana', 'Haiti', 'Honduras',
    'Hungary', 'India', 'Indonesia', 'Iraq', 'Ireland', 'Italy',
    'Jamaica', 'Japan', 'Kazakhstan', 'Kenya', 'Latvia', 'Lebanon',
    'Lesotho', 'Libya', 'Lithuania', 'Madagascar', 'Malawi',
    'Malaysia', 'Mali', 'Mauritania', 'Mauritius', 'Mexico',
    'Montenegro', 'Morocco', 'Mozambique', 'Namibia', 'Nepal',
    'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Norway',
    'Pakistan', 'Papua New Guinea', 'Peru', 'Poland', 'Portugal',
    'Qatar', 'Romania', 'Rwanda', 'Saudi Arabia', 'Senegal',
    'Slovenia', 'South Africa', 'Spain', 'Sri Lanka', 'Sudan',
    'Suriname', 'Sweden', 'Switzerland', 'Tajikistan', 'Thailand',
    'Tunisia', 'Turkey', 'Uganda', 'Ukraine', 'United Kingdom',
    'Uruguay', 'Zambia', 'Zimbabwe'
]

CROP_LIST = [
    'Maize', 'Potatoes', 'Rice, paddy', 'Sorghum', 'Soybeans', 'Wheat',
    'Cassava', 'Sweet potatoes', 'Plantains and others', 'Yams'
]

# ---------------- INPUT FORM ----------------
with st.form("yield_input_form"):

    st.subheader("📌 Crop & Location Details")

    area = st.selectbox(
        label="Area / Country",
        options=AREA_LIST
        
    )

    item = st.selectbox(
        label="Crop Type",
        options=CROP_LIST
    )

    year = st.number_input(
        label="Year",
        min_value=1960,
        max_value=2050,
        value=2013,
        step=1
    )

    st.subheader("🌦 Environmental Factors")

    rainfall = st.number_input(
        label="Average Rainfall (mm/year)",
        min_value=0.0,
        value=1000.0
    )

    pesticides = st.number_input(
        label="Pesticides Used (tonnes)",
        min_value=0.0,
        value=30000.0
    )

    temperature = st.number_input(
        label="Average Temperature (°C)",
        min_value=-10.0,
        max_value=60.0,
        value=25.0
    )

    submit = st.form_submit_button("🚀 Submit")

# ---------------- INPUT DATAFRAME ----------------
if submit:
    input_df = pd.DataFrame([{
        "Area": area,
        "Item": item,
        "Year": year,
        "average_rain_fall_mm_per_year": rainfall,
        "pesticides_tonnes": pesticides,
        "avg_temp": temperature
    }])

    st.success("Input data captured successfully ✅")
    
    with open('artifacts/Age_encoder.pkl','rb') as File:
        label_encoder = pickle.load(File)
    
    with open('artifacts/onehot_encoder.pkl','rb') as File:
        onehot_encode = pickle.load(File) 
    
    with open('artifacts/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
        
    ## Transform item
    Item_encoder = onehot_encode.transform(input_df[['Item']]).toarray()
    onehot_encode.get_feature_names_out(['Item'])
    Item_encoded_df=pd.DataFrame(Item_encoder,columns=onehot_encode.get_feature_names_out(['Item']))
    input_df=pd.concat([input_df.drop('Item',axis=1),Item_encoded_df],axis=1)
    
    ## Transform Area
    input_df["Area_te"] = label_encoder.transform(input_df["Area"])
    input_df.drop(columns=["Area"], inplace=True)
    
    ## Scaling the input data
    input_scaled=scaler.transform(input_df)
    
    
    ## Final Prediction
    predict = model.predict(input_scaled)
    prediction = predict[0][0]
    
    
    
    st.write(f"Crop yield in hectograms per hectare: {prediction} hg/ha")
    st.dataframe(input_df)

