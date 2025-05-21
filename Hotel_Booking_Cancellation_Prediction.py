import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
import pickle

# Judul Aplikasi
st.title("🏨 Hotel Booking Cancellation Prediction")
st.markdown("""
Welcome to the hotel booking prediction tool!  
This app uses **Machine Learning** to predict whether a hotel booking will be **canceled** or **honored** based on customer behavior and booking details.  
Use the sidebar to enter the booking features and get a **real-time prediction**.
""")

st.markdown("---")

# Sidebar Input
st.sidebar.header("📋 Input Booking Details Below")

def input_user():
    country = st.sidebar.selectbox("🌍 Country", ['IRL', 'FRA', 'PRT', 'NLD', 'ESP', 'UMI', 'CN', 'LUX', 'BRA', 'BEL', 'JPN', 'DEU', 'ITA', 'CHE', 'GBR', 'AGO', 'SRB', 'COL', 'CHN', 'SWE', 'AUT', 'CIV', 'CZE', 'POL', 'USA', 'SGP', 'RUS', 'ROU', 'DNK', 'IND', 'MAR', 'PHL', 'ARG', 'ISL', 'ZAF', 'LBN', 'MOZ', 'TUR', 'BGD', 'MEX', 'CAF', 'NOR', 'FIN', 'UKR', 'EGY', 'ISR', 'Unknown', 'KOR', 'AZE', 'HUN', 'AUS', 'EST', 'CHL', 'SVN', 'PRY', 'ABW', 'ALB', 'LTU', 'ARE', 'HRV', 'SAU', 'NZL', 'LVA', 'ATA', 'KAZ', 'DZA', 'TWN', 'CRI', 'BIH', 'BGR', 'IRQ', 'OMN', 'VEN', 'IDN', 'GEO', 'MLT', 'IRN', 'BLR', 'URY', 'LBY', 'TUN', 'BEN', 'MYS', 'MWI', 'GRC', 'CYP', 'CPV', 'HKG', 'PRI', 'MKD', 'MUS', 'IMN', 'PAN', 'NGA', 'GLP', 'KHM', 'PER', 'QAT', 'SEN', 'MAC', 'SVK', 'BHR', 'ECU', 'SYC', 'BOL', 'TJK', 'LCA', 'MDV', 'SYR', 'ZMB', 'LIE', 'THA', 'MNE', 'BRB', 'CMR', 'JEY', 'GTM', 'LKA', 'JOR', 'TZA', 'AND', 'ARM', 'GIB', 'VNM', 'PAK', 'JAM', 'DOM', 'KWT', 'LAO', 'RWA', 'FRO', 'GAB', 'ETH', 'CUB', 'COM', 'GNB', 'GGY', 'NIC', 'TGO', 'TMP', 'CYM', 'GHA', 'SLE', 'BWA', 'NCL', 'UZB', 'SUR', 'SDN', 'PLW', 'MLI', 'MMR', 'BFA', 'SLV', 'BDI', 'ZWE', 'UGA', 'DMA', 'VGB', 'KIR', 'KEN', 'MYT', 'KNA', 'AIA'])
    market_segment = st.sidebar.selectbox("🛒 Market Segment", ['Online TA', 'Offline TA/TO', 'Groups', 'Direct', 'Corporate', 'Complementary', 'Aviation', 'Undefined'])
    previous_cancellations = st.sidebar.number_input("❌ Previous Cancellations", min_value=0, max_value=26, value=0, step=1)
    booking_changes = st.sidebar.number_input("🔄 Booking Changes", min_value=0, max_value=26, value=0, step=1)
    deposit_type = st.sidebar.selectbox("💳 Deposit Type", ['No Deposit', 'Non Refund', 'Refundable'])
    days_in_waiting_list = st.sidebar.number_input("🕒 Days in Waiting List", min_value=0, max_value=391, value=0, step=1)
    customer_type = st.sidebar.radio("👤 Customer Type", ['Transient', 'Contract', 'Group', 'Transient-Party'])
    reserved_room_type = st.sidebar.selectbox("🛏️ Reserved Room Type", ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'P', 'L'])
    required_car_parking_spaces = st.sidebar.number_input("🚗 Required Car Parking Spaces", min_value=0, max_value=8, value=0, step=1)
    total_of_special_requests = st.sidebar.number_input("⭐ Total of Special Requests", min_value=0, max_value=5, value=0, step=1)

    df = pd.DataFrame({
        "country": [country],
        "market_segment": [market_segment],
        "previous_cancellations": [previous_cancellations],
        "booking_changes": [booking_changes],
        "deposit_type": [deposit_type],
        "days_in_waiting_list": [days_in_waiting_list],
        "customer_type": [customer_type],
        "reserved_room_type": [reserved_room_type],
        "required_car_parking_spaces": [required_car_parking_spaces],
        "total_of_special_requests": [total_of_special_requests]
    })

    return df

# Ambil data dari input sidebar
df_hotel_booking = input_user()
df_hotel_booking.index = [0]

st.markdown("---")
st.subheader("📑 Booking Summary")
st.write(df_hotel_booking)

# Load model
model_loaded = pickle.load(open("model_gradient.sav", "rb"))
kelas = model_loaded.predict(df_hotel_booking)

# Output prediksi
st.markdown("---")
st.subheader("🔍 Prediction Result")

if kelas == 0:
    st.success("✅ **Prediction: Booking will be KEPT**")
    st.info("This guest is **likely to arrive as scheduled**. No further action needed.")
else:
    st.error("⚠️ **Prediction: Booking will be CANCELED**")
    st.warning("This guest is **likely to cancel the booking**. Consider contacting them or applying your overbooking strategy.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Scikit-Learn")
