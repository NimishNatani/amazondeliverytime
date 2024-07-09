import streamlit as st
import numpy as np
import haversine as hs   
from haversine import Unit
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

svm_model_path = 'svm_model.pkl'
with open(svm_model_path, 'rb') as file:
    svm_model = pickle.load(file)

# voting_model_path = 'voting_model.pkl'
# with open(voting_model_path, 'rb') as file:
#     voting_model = pickle.load(file)
df=pd.read_csv('updated-amazon-time.csv')

def predict_speed(model, features):
    return model.predict(features)

def calculate_travel_time(distance, speed):
    return (distance / speed) * 60

def calculate_distance(pickup_lat, pickup_long, drop_lat, drop_long):
    return hs.haversine((pickup_lat, pickup_long), (drop_lat,drop_long), unit=Unit.KILOMETERS)

st.title('Travel Time Predictor')

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

    speed1 = predict_speed(svm_model, input_df)

    time1 = calculate_travel_time(distance, speed1)

    time1 = np.array(time1).item()

    st.header(f'Predicted travel time range: {(time1-10):.2f} to {(time1 + 5):.2f} minutes')
    st.header("Analysis based on similar conditions from previous deliveries")
    filtered_user_data = df[
        (df['Distance_KM'] >= (distance - 2)) & (df['Distance_KM'] <= (distance + 2)) &
        (df['Agent_Age'] >= (agent_age - 2)) & (df['Agent_Age'] <= (agent_age + 2)) &
        (df['Agent_Rating'] >= (agent_rating - 0.3)) & (df['Agent_Rating'] <= (agent_rating + 0.3)) &
        (df['Weather'] == weather) &
        (df['Traffic'] == traffic) &
        (df['Vehicle'] == vehicle) &
        (df['Area'] == area)
    ]
    if not filtered_user_data.empty:
    
           mean_travel_time = filtered_user_data['Delivery_Time'].mean()
           st.write(f'Mean travel time based on previous deliveries: {mean_travel_time:.2f} minutes')
           col5,col6 = st.columns(2)
           with col5:
               fig, ax = plt.subplots()
               sns.histplot(filtered_user_data['Delivery_Time'], kde=True, bins=20, ax=ax)
               ax.set_title('Distribution of Travel Times')
               ax.set_xlabel('Travel Time (minutes)')
               ax.set_ylabel('Frequency')
               st.pyplot(fig)
           
           with col6:
               fig, ax = plt.subplots()
               scatter = ax.scatter(filtered_user_data['Distance_KM'], filtered_user_data['Delivery_Time'], c=filtered_user_data['Agent_Rating'], cmap='viridis')
               ax.set_title('Distance vs Travel Time')
               ax.set_xlabel('Distance (km)')
               ax.set_ylabel('Travel Time (minutes)')
               fig.colorbar(scatter, label='Agent Rating')
               st.pyplot(fig)
    else :
        st.write("No similar data found based on current conditions")
   
else:
    st.write("Please enter the distance or calculate it using the latitude and longitude.")
