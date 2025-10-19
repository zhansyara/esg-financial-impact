import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.title("ESG–Financial Impact Predictor")
st.write("Input your financial indicators to estimate EBITDA Margin and visualize ESG impact.")

# Load trained model (after saving it)
import joblib
model = joblib.load("/Users/zhansaya/Downloads/ebitda_model.pkl")

# model = joblib.load("ebitda_model.pkl")

# User inputs
esg = st.slider("ESG Rating Score", 0, 100, 50)
pe = st.number_input("P/E Ratio", 0.0)
revenue = st.number_input("Revenue", 0.0)
ebitda = st.number_input("EBITDA", 0.0)
net_income = st.number_input("Net Income", 0.0)
eps = st.number_input("EPS", 0.0)
market_cap = st.number_input("Market Cap", 0.0)
debt_to_equity = st.number_input("Debt to Equity", 0.0)
current_ratio = st.number_input("Current Ratio", 0.0)
quick_ratio = st.number_input("Quick Ratio", 0.0)
return1y = st.number_input("1Y Return", 0.0)
volatility = st.number_input("Avg Volatility (30D)", 0.0)

# When user clicks
if st.button("Predict EBITDA Margin"):
    input_data = pd.DataFrame([[
        esg, pe, revenue, ebitda, net_income, eps, market_cap,
        debt_to_equity, current_ratio, quick_ratio, return1y, volatility
    ]], columns = [
    "ESG_Rating_Score", "P/E Ratio", "Revenue", "EBITDA",
    "Net Income", "EPS", "Market Cap", "Debt to Equity",
    "Current Ratio", "Quick Ratio", "1Y Return", "Avg Volatility (30D)"
])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted EBITDA Margin: {round(prediction, 2)}")

    # ESG impact curve
    esg_values = np.linspace(0, 100, 50)
    predicted_margins = []
    for e in esg_values:
        temp = input_data.copy()
        temp["ESG_Rating_Score"] = e
        predicted_margins.append(model.predict(temp)[0])

    plt.figure()
    plt.plot(esg_values, predicted_margins)
    plt.xlabel("ESG Rating Score")
    plt.ylabel("Predicted EBITDA Margin")
    plt.title("How ESG Score Affects EBITDA Margin")
    st.pyplot(plt)
