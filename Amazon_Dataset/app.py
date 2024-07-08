import streamlit as st
import numpy as np
import haversine as hs   
from haversine import Unit
import pickle
import pandas as pd

svm_model_path = 'svm_model.pkl'
with open(svm_model_path, 'rb') as file:
    svm_model = pickle.load(file)

# voting_model_path = 'voting_model.pkl'
# with open(voting_model_path, 'rb') as file:
#     voting_model = pickle.load(file)

def predict_speed(model, features):
    return model.predict(features)

def calculate_travel_time(distance, speed):
    return (distance / speed) * 60

def calculate_distance(pickup_lat, pickup_long, drop_lat, drop_long):
    return hs.haversine((pickup_lat, pickup_long), (drop_lat,drop_long), unit=Unit.KILOMETERS)

# Streamlit app
st.title('Travel Time Predictor')

# User input for method of distance entry
distance_method = st.radio('How would you like to enter the distance?', ('Latitude/Longitude', 'Distance in Kilometers'))

if distance_method == 'Latitude/Longitude':
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        pickup_lat = st.number_input('Pickup Latitude', value=0.0, format="%.6f")
    with col2:
        pickup_long = st.number_input('Pickup Longitude', value=0.0, format="%.6f")
    with col3:
        drop_lat = st.number_input('Drop Latitude', value=0.0, format="%.6f")
    with col4:
        drop_long = st.number_input('Drop Longitude', value=0.0, format="%.6f")
    if st.button('Calculate Distance'):
        distance = calculate_distance(pickup_lat, pickup_long, drop_lat, drop_long)
        st.write(f'Calculated Distance: {distance:.2f} km')
    else:
        distance = None
else:
    distance = st.number_input('Distance (km)', value=0.0)

agent_age = st.number_input('Agent Age', value=25)
agent_rating = st.slider('Agent Rating', 1.0, 5.0, step=0.1)
weather = st.selectbox('Weather', ['Sunny', 'Cloudy', 'Sandstorms', 'Windy', 'Fog','Stormy'])
vehicle = st.selectbox('Vehicle Type', ['scooter', 'motorcycle'])
area = st.selectbox('Area Type', ['Urban', 'Metropolitian', 'Other'])
traffic = st.selectbox('Traffic Conditions', ['Low', 'Medium', 'High', 'Jam'])

# Prepare feature vector
if st.button("Calculate Time "):
    input_df = pd.DataFrame({
        'Agent_Age': [agent_age],
        'Agent_Rating': [agent_rating],
        'Distance_KM': [distance],
        'Weather': [weather],
        'Traffic': [traffic],
        'Vehicle': [vehicle],
        'Area': [area]
    })

    # Predict speed using both models
    speed1 = predict_speed(svm_model, input_df)
    # speed2 = predict_speed(voting_model, input_df)

    # Calculate travel time in minutes
    time1 = calculate_travel_time(distance, speed1)
    # time2 = calculate_travel_time(distance, speed2)

    # Display results
    time1 = np.array(time1).item()
    # time2 = np.array(time2).item()

        # Display results with ±5 minutes buffer
    st.write(f'Predicted travel time range: {(time1-10):.2f} to {(time1 + 5):.2f} minutes')
   
else:
    st.write("Please enter the distance or calculate it using the latitude and longitude.")