import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import pandas as pd
from PIL import Image


def run():
    st.title("EaseMyTrip Flight Dataset")

    image = Image.open("./res/header_img.jpg")
    st.image(image, use_column_width=True)

    df = pd.read_csv(r"././data/Flight_Price_Prediction.csv")

    st.dataframe(df)

    pr = ProfileReport(df, title="Profile Report", minimal=True)

    st_profile_report(pr)



if __name__ == "__main__":
    run()
