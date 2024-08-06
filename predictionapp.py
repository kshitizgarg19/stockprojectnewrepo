import streamlit as st
from datetime import date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from plotly import graph_objs as go
from sklearn.preprocessing import StandardScaler

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Function to get the next business day
def next_business_day(date):
    next_day = date + timedelta(days=1)
    while next_day.weekday() >= 5:  # Skip Saturday (5) and Sunday (6)
        next_day += timedelta(days=1)
    return next_day


import streamlit as st

# HTML/CSS for centered and bold title with dark theme
title_style = """
<style>
/* Center and style the title */
.title {
  text-align: center;
  font-weight: bold;
  font-size: 2.5em; /* Adjust font size as needed */
  margin-bottom: 1em; /* Add some space below the title */
}

/* Dark theme */
body {
    background-color: #2d3436; /* Dark gray background */
    color: white; /* White text */
}
</style>
"""

# Render the title and apply dark theme
st.markdown(title_style, unsafe_allow_html=True)
st.markdown("<div class='title'>Stock Price prediction </div>", unsafe_allow_html=True)

# Summary section
st.subheader("About Stock Price prediction ")

st.write("""
Stock Price Prediction empower investors by providing precise predictions of stock prices 
through advanced machine learning algorithms. Our platform leverages state-of-the-art 
techniques to analyze historical data, forecast trends, and assist users in making 
informed investment decisions. 
""")

markets = ['US', 'India']
market = st.radio('Select market:', markets)

if market == 'US':
    options = {
        'Apple (AAPL)': 'AAPL',
        'Microsoft (MSFT)': 'MSFT',
        'Amazon (AMZN)': 'AMZN',
        'Google (GOOGL)': 'GOOGL',
        'Tesla (TSLA)': 'TSLA',
        'S&P 500 ETF (SPY)': 'SPY',  # Example of US index fund
    }
    currency = '$'
else:
    options = {
        'Reliance Industries (RELIANCE.NS)': 'RELIANCE.NS',
        'Tata Consultancy Services (TCS.NS)': 'TCS.NS',
        'HDFC Bank (HDFCBANK.NS)': 'HDFCBANK.NS',
        'ICICI Bank (ICICIBANK.NS)': 'ICICIBANK.NS',
        'Hindustan Unilever (HINDUNILVR.NS)': 'HINDUNILVR.NS',
        'ITC Ltd (ITC.NS)': 'ITC.NS',
        'State Bank of India (SBIN.NS)': 'SBIN.NS',
        'Axis Bank (AXISBANK.NS)': 'AXISBANK.NS',
        'Bharti Airtel (BHARTIARTL.NS)': 'BHARTIARTL.NS',
        'Asian Paints (ASIANPAINT.NS)': 'ASIANPAINT.NS',
        'Titan Company (TITAN.NS)': 'TITAN.NS',
        'Nestle India (NESTLEIND.NS)': 'NESTLEIND.NS',
        'HDFC Life Insurance (HDFCLIFE.NS)': 'HDFCLIFE.NS',
        'Bajaj Finance (BAJFINANCE.NS)': 'BAJFINANCE.NS',
        'Bajaj Finserv (BAJAJFINSV.NS)': 'BAJAJFINSV.NS',
        'Maruti Suzuki India (MARUTI.NS)': 'MARUTI.NS',
        'UPL Ltd (UPL.NS)': 'UPL.NS',
        'Wipro Ltd (WIPRO.NS)': 'WIPRO.NS',
        'Coal India Ltd (COALINDIA.NS)': 'COALINDIA.NS',
        'Larsen & Toubro Ltd (LT.NS)': 'LT.NS',
        'Tech Mahindra Ltd (TECHM.NS)': 'TECHM.NS',
        'HCL Technologies (HCLTECH.NS)': 'HCLTECH.NS',
        'Sun Pharmaceutical Industries (SUNPHARMA.NS)': 'SUNPHARMA.NS',
        'Power Grid Corporation of India (POWERGRID.NS)': 'POWERGRID.NS',
        'NTPC Ltd (NTPC.NS)': 'NTPC.NS',
        'Hindalco Industries (HINDALCO.NS)': 'HINDALCO.NS',
        'Tata Steel Ltd (TATASTEEL.NS)': 'TATASTEEL.NS',
        'Adani Ports & SEZ (ADANIPORTS.NS)': 'ADANIPORTS.NS',
        'Shree Cement Ltd (SHREECEM.NS)': 'SHREECEM.NS',
        'IndusInd Bank Ltd (INDUSINDBK.NS)': 'INDUSINDBK.NS',
        'Britannia Industries (BRITANNIA.NS)': 'BRITANNIA.NS',
        'Hero MotoCorp Ltd (HEROMOTOCO.NS)': 'HEROMOTOCO.NS',
        'ICICI Prudential Life Insurance (ICICIPRULI.NS)': 'ICICIPRULI.NS',
        'Dr. Reddy\'s Laboratories (DRREDDY.NS)': 'DRREDDY.NS',
        'Mahindra & Mahindra Ltd (M&M.NS)': 'M&M.NS',
        'UltraTech Cement Ltd (ULTRACEMCO.NS)': 'ULTRACEMCO.NS',
        'Divi\'s Laboratories Ltd (DIVISLAB.NS)': 'DIVISLAB.NS',
        'Cipla Ltd (CIPLA.NS)': 'CIPLA.NS',
        'Eicher Motors Ltd (EICHERMOT.NS)': 'EICHERMOT.NS',
        'SBI Life Insurance Company Ltd (SBILIFE.NS)': 'SBILIFE.NS',
        'Pidilite Industries Ltd (PIDILITIND.NS)': 'PIDILITIND.NS',
        'Godrej Consumer Products Ltd (GODREJCP.NS)': 'GODREJCP.NS',
        'Bank Nifty (BANKNIFTY.NS)': '^NSEBANK',  # Example of Indian index fund (Bank Nifty)
        'Nifty 50 (NIFTY.NS)': '^NSEI',  # Example of Indian index fund (Nifty 50)
        'Sensex (SENSEX.BSE)': '^BSESN'  # Example of Indian index (Sensex)
        # Add more Indian index funds as needed
    }
    currency = '₹'

selected_stock = st.selectbox('Select dataset for prediction', list(options.keys()))
ticker = options[selected_stock]

custom_ticker = st.text_input('Or enter a custom ticker (optional):')
if custom_ticker:
    ticker = custom_ticker

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(ticker)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Check if df_train has valid data
if len(data) == 0:
    st.error('No data available for the selected stock. Please choose another stock or check your custom ticker.')
else:
    # Plot raw data
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True, xaxis_title='Date', yaxis_title=f'Price ({currency})')
        st.plotly_chart(fig)
        
    plot_raw_data()

    # Prepare the data for gradient boosting regression
    df_train = data[['Date', 'Close']]
    df_train['Date'] = pd.to_datetime(df_train['Date'])
    df_train['Date_ordinal'] = df_train['Date'].map(pd.Timestamp.toordinal)

    if len(df_train) == 0:
        st.error('No valid data available for the selected stock. Please choose another stock or check your custom ticker.')
    else:
        X = df_train[['Date_ordinal']]
        y = df_train['Close']

        # Scaling the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit the model
        model = GradientBoostingRegressor()
        model.fit(X_scaled, y)

        # Predict next business day's close price
        next_day = next_business_day(pd.to_datetime(TODAY))
        next_day_ordinal = np.array([[next_day.toordinal()]])
        next_day_scaled = scaler.transform(next_day_ordinal)
        next_day_prediction = model.predict(next_day_scaled)[0]

        # Create a DataFrame for the forecast
        forecast = pd.DataFrame({'Date': [next_day], 'Predicted Close': [next_day_prediction]})

        # Show and plot forecast
        st.subheader('Forecast data')
        st.write(forecast)

        # Plot the forecast data
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df_train['Date'], y=df_train['Close'], name="Actual Close"))
        fig1.add_trace(go.Scatter(x=forecast['Date'], y=forecast['Predicted Close'], name="Predicted Close", line=dict(color='royalblue')))
        fig1.layout.update(title_text='Forecast plot', xaxis_rangeslider_visible=True, xaxis_title='Date', yaxis_title=f'Price ({currency})')
        st.plotly_chart(fig1)

        # Components plot (simple plot of the regression line)
        st.write("Forecast components")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_train['Date'], y=df_train['Close'], mode='markers', name='Actual'))
        fig2.add_trace(go.Scatter(x=forecast['Date'], y=forecast['Predicted Close'], mode='lines', name='Forecast', line=dict(color='royalblue')))
        fig2.layout.update(title_text='Forecast components', xaxis_title='Date', yaxis_title=f'Price ({currency})')
        st.plotly_chart(fig2)

        # Display the predicted price
        st.subheader('Predicted Price')
        st.markdown(f"<div style='text-align: center; font-size: 24px;'>Predicted Close Price on {next_day.strftime('%A, %d-%m-%Y')}: {currency}{next_day_prediction:.2f}</div>", unsafe_allow_html=True)

# Developer attribution
# Developer attribution and footer
# Developer attribution
# Developer attribution aligned to the right
# Developer attribution aligned to the right with icons
# Developer attribution with clickable links
st.markdown("""
    <br><br><br>
    <div style="text-align: right;">
       <p style="font-weight: bold;">Developed and maintained by Kshitiz Garg</p> 
       <p>GitHub: <a href="https://github.com/kshitizgarg19">GitHub</a></p>
       <p>LinkedIn: <a href="https://www.linkedin.com/in/kshitiz-garg-898403207/">LinkedIn</a></p>
       <p>Instagram: <a href="https://www.instagram.com/kshitiz_garg_19?igsh=aWVjaGE0NThubG80&utm_source=qr">Instagram</a></p>
       <p>WhatsApp: <a href="https://wa.me/918307378790">Chat on WhatsApp</a></p>
       <p>Email: <a href="mailto:kshitizgarg19@gmail.com">kshitizgarg19@gmail.com</a></p>
    </div>
""", unsafe_allow_html=True)




