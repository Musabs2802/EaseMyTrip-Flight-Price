import streamlit as st
import pandas as pd
from PIL import Image

def run():
    st.title("EaseMyTrip Flight Dataset Insights")

    image = Image.open("./res/header_img.jpg")
    st.image(image, use_column_width=True)

    df = pd.read_csv(r"././data/Flight_Price_Prediction.csv")
    
    st.header("What is the frequency of each flights in each city?")
    flight_counts = df.groupby(['source_city', 'airline'])['airline'].count().unstack()
    st.bar_chart(flight_counts)
    st.write("- Vistara has the highest number of departures in every city, followed by Air India.")
    st.write("- SpiceJet has the least number of departure flights from any citites.")
    

    st.header("Does price vary with airlines?")
    airline_mean_prices = df.groupby(['airline', 'class'])['price'].mean().unstack()
    st.bar_chart(airline_mean_prices)
    st.write("- Business class only exists for Air India and Vistara.")
    st.write("- Business class mean prices for Air India and Vistara are the highest. This also explains why Air India and Vistara have the highest number of departures in any city.")
    st.write("- A conclusion can be made that these highest number of departures from Air India and Vistara are due to businessman travelling between these cities.")
    

    st.header("How is the price affected when booked over the days before the flight?")
    days_left_prices = df.groupby('days_left')['price'].mean()
    st.line_chart(days_left_prices)
    st.write("- It is observed that as one moves closer to the flight time, flight prices increases heavily.")
    st.write("- It is also observed that the prices are much cheaper if booked as early as possible.")


    st.header("Does the ticket price change based on the departure time and arrival time?")
    departure_time_prices = df.groupby(['departure_time', 'airline'])['price'].mean().unstack()
    st.bar_chart(departure_time_prices)
    st.write("- SpiceJet and Vistara donot operate on late night")
    st.write("- Late Night flights are generally cheaper")
    st.write("- Air India Late Night flights are expensive")
    st.write("- Morning and Afternoon flights are generally relatively expensive")


    st.header("Does prices depend on source city ?")
    source_city_prices = df.groupby('source_city')['price'].mean()
    st.bar_chart(source_city_prices)
    destination_city_prices = df.groupby('destination_city')['price'].mean()
    st.bar_chart(destination_city_prices)
    st.write("- From above analysis, it is observed that price does not significantly depend on both source and destination.")

if __name__ == '__main__':
    run()