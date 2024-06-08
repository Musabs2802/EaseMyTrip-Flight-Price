import streamlit as st
import joblib
import statsmodels.api as sm
import pandas as pd
import pickle
from PIL import Image

model = joblib.load('./output/model.sav')
df = pd.read_csv(r"./data/Flight_Price_Prediction.csv")

with open('./output/airline_map.pkl', 'rb') as f:
    airline_map = pickle.load(f)

with open('./output/arrival_time_map.pkl', 'rb') as f:
    arrival_time_map = pickle.load(f)

with open('./output/class_map.pkl', 'rb') as f:
    class_map = pickle.load(f)

with open('./output/departure_time_map.pkl', 'rb') as f:
    departure_time_map = pickle.load(f)

with open('./output/stops_map.pkl', 'rb') as f:
    stops_map = pickle.load(f)

cities = ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai']

st.cache_data.clear()

def run():
    st.title("EaseMyTrip Flight Prediction")
    
    image = Image.open("./res/header_img.jpg")
    st.image(image, use_column_width=True)

    select_model = st.sidebar.selectbox("Choose Model:", ("OLS Linear Regression", "Neural Network"))
    
    if select_model == 'OLS Linear Regression':
        st.sidebar.info("This app predicts price of your next flight")
        model = joblib.load('./output/model.sav')
    
        col1, col2 = st.columns(2)
        airline = col1.selectbox("Airline", airline_map.values())

        # stops = col2.selectbox("Stops", stops_map.values())
        stops = col2.selectbox("Stops", df[df['airline']==airline]['stops'].unique())

        # source_city = col1.selectbox("Source City", cities)
        source_city = col1.selectbox("Source City", df[(df['airline']==airline) & (df['stops']==stops)]['source_city'].unique())

        destination_city = col2.selectbox("Destination City", df[(df['airline']==airline) & (df['stops']==stops) & (df['source_city']==source_city)]['destination_city'].unique())
        
        departure_time = col1.selectbox("Departure Time", df[(df['airline']==airline) & (df['stops']==stops)]['departure_time'].unique())
        arrival_time = col2.selectbox("Arrival Time", df[(df['airline']==airline) & (df['stops']==stops) & (df['departure_time']==departure_time)]['arrival_time'].unique())

        class_ = col1.selectbox("Class", class_map.values())

        duration = st.slider("Duration", min_value=df[(df['stops']==stops) & (df['departure_time']==departure_time) & (df['arrival_time']==arrival_time)]['duration'].min(), max_value=df[(df['stops']==stops) & (df['departure_time']==departure_time) & (df['arrival_time']==arrival_time)]['duration'].max(), step=0.1)
        days_left = col1.slider("Days Before", min_value=df[(df['airline']==airline) & (df['stops']==stops) & (df['departure_time']==departure_time) & (df['arrival_time']==arrival_time)]['days_left'].min(), max_value=df[(df['airline']==airline) & (df['stops']==stops) & (df['departure_time']==departure_time) & (df['arrival_time']==arrival_time)]['days_left'].max())

        data = pd.DataFrame(
            [
                {
                    "airline": list(airline_map.keys())[list(airline_map.values()).index(airline)],
                    "departure_time": list(departure_time_map.keys())[list(departure_time_map.values()).index(departure_time)],
                    "stops": list(stops_map.keys())[list(stops_map.values()).index(stops)],
                    "arrival_time": list(arrival_time_map.keys())[list(arrival_time_map.values()).index(arrival_time)],
                    "class": list(class_map.keys())[list(class_map.values()).index(class_)],
                    "duration": duration,
                    "days_left": days_left,
                }
            ]
        )

        data = sm.add_constant(data, has_constant='add')

        output = ""
        if st.button("Predict", type="primary"):
            output = model.predict(data)
            output = max(0, output.tolist()[0])

        if output is not "":
            st.info("Estimated Price: {}".format(round(output*1000, 2)))
    
    if select_model=="Neural Network":
        st.sidebar.warning("This option is currently disabled. Please select another option.")

if __name__ == "__main__":
    run()