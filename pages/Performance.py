import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
from PIL import Image

model = joblib.load('./output/model.sav')

def run():
    with open('./output/df_encoded.pkl', 'rb') as f:
        df_encoded = pickle.load(f)

    st.title("EaseMyTrip Models Performance")

    image = Image.open("./res/header_img.jpg")
    st.image(image, use_column_width=True)

    select_model = st.sidebar.selectbox("Choose Model:", ("OLS Linear Regression", "Neural Network"))

    if select_model=="OLS Linear Regression":
        st.sidebar.info("This gives a summary of the selected model")
        model = joblib.load('./output/model.sav')
        st.text(model.summary())

        df_encoded = df_encoded.sample(frac=1).reset_index(drop=True)
        split_index = int(len(df_encoded) * 0.8)
        test_df = df_encoded[split_index:]
        x_test = test_df.drop('price', axis=1)
        y_test = test_df['price']/1000

        x_test_with_intercept = sm.add_constant(x_test)
        y_pred = model.predict(x_test_with_intercept)

        # Evaluate the model
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
        ss_residual = np.sum((y_test - y_pred) ** 2)
        r_squared = 1 - (ss_residual / ss_total)

        st.subheader("Metrics on test data")
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"Root Mean Squared Error: {rmse}")
        st.write(f"Mean Absolute Error: {mae}")
        st.write(f"R-squared: {r_squared}")

        scatter_data = pd.DataFrame({
            'Actual Values': y_test[:500],
            'Predicted Values': y_pred[:500]
        })
        st.line_chart(scatter_data, color=["#0000FF", "#FF0000"])

    if select_model=="Neural Network":
        st.sidebar.warning("This option is currently disabled. Please select another option.")
        

if __name__ == "__main__":
    run()
