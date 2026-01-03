# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# -------------------------
# Load dataset and train model
# -------------------------
df = pd.read_csv("data.csv")
df = df.rename(columns={'Date':'day','Month':'month','Year':'year'})
df['Date_Combined'] = pd.to_datetime(df[['year','month','day']])
df['DayOfYear'] = df['Date_Combined'].dt.dayofyear
df['Month'] = df['Date_Combined'].dt.month
df['Year'] = df['Date_Combined'].dt.year

features = ['Holidays_Count','Days','PM2.5','PM10','NO2','SO2','CO','Ozone','DayOfYear','Month','Year']
X = df[features]
y = df['AQI']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_scaled, y)

# -------------------------
# Function to categorize AQI
# -------------------------
def aqi_category(aqi):
    if aqi <= 50:
        return "Good", "green", "Air quality is satisfactory."
    elif aqi <= 100:
        return "Moderate", "yellow", "Air quality is acceptable."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "orange", "People with health issues should limit outdoor activities."
    elif aqi <= 200:
        return "Unhealthy", "red", "Everyone may experience health effects. Limit outdoor activities."
    elif aqi <= 300:
        return "Very Unhealthy", "purple", "Health alert! Everyone may experience more serious health effects."
    else:
        return "Hazardous", "maroon", "Health warnings of emergency conditions. Stay indoors."

# -------------------------
# Streamlit Interface
# -------------------------
st.set_page_config(page_title="Air Quality Prediction", layout="centered")
st.title("🌫️ Air Quality Prediction System")

with st.form(key='aqi_form'):
    date = st.date_input("Date")
    holidays_count = st.number_input("Holidays Count", min_value=0, value=0)
    days = st.number_input("Day of the Week (1=Monday,...7=Sunday)", min_value=1, max_value=7, value=1)
    pm25 = st.number_input("PM2.5", min_value=0.0, value=50.0)
    pm10 = st.number_input("PM10", min_value=0.0, value=60.0)
    no2 = st.number_input("NO2", min_value=0.0, value=20.0)
    so2 = st.number_input("SO2", min_value=0.0, value=10.0)
    co = st.number_input("CO", min_value=0.0, value=0.5)
    ozone = st.number_input("Ozone", min_value=0.0, value=30.0)
    
    submit_button = st.form_submit_button(label="Predict AQI")

if submit_button:
    day_of_year = date.timetuple().tm_yday
    month = date.month
    year = date.year
    
    user_input = {
        'Holidays_Count': holidays_count,
        'Days': days,
        'PM2.5': pm25,
        'PM10': pm10,
        'NO2': no2,
        'SO2': so2,
        'CO': co,
        'Ozone': ozone,
        'DayOfYear': day_of_year,
        'Month': month,
        'Year': year
    }
    
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    predicted_aqi = model.predict(input_scaled)[0]
    level, color, suggestion = aqi_category(predicted_aqi)
    
    # Display results
    st.subheader(f"Predicted AQI: {predicted_aqi:.2f}")
    st.subheader(f"Air Quality Level: {level}")
    st.markdown(f"**Suggestion:** {suggestion}")
    
    # Show user input table
    st.subheader("User Input Parameters")
    st.table(input_df.T.rename(columns={0:"Value"}))
    
    # Bar chart
    fig, ax = plt.subplots()
    ax.bar(["Predicted AQI"], [predicted_aqi], color=color)
    ax.set_ylim(0, max(500, predicted_aqi+50))
    ax.set_ylabel("AQI")
    ax.set_title("Predicted AQI")
    st.pyplot(fig)
