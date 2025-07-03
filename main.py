import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import cv2  # <-- Add OpenCV
import requests
from bs4 import BeautifulSoup

sns.set(style="whitegrid")

def download_data(ticker='DIVISLAB.NS', start='2020-01-01', end='2023-01-01'):
    df = yf.download(ticker, start=start, end=end)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

def preprocess_data(df):
    df = df.copy()
    df = df.ffill().bfill()
    return df

def plot_eda_interactive(df, ticker, start_date, end_date):
    # Filter by selected date range
    df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

    # 1. Close Price (Interactive)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
    fig1.update_layout(
        title='Close Price',
        xaxis_title='Date',
        yaxis_title='Close',
        height=350,
        xaxis=dict(rangeslider=dict(visible=True), type="date")
    )
    st.plotly_chart(fig1, use_container_width=True)

    # 2. Rolling Mean & Std (Interactive)
    rolling_mean = df['Close'].rolling(30).mean()
    rolling_std = df['Close'].rolling(30).std()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Original'))
    fig2.add_trace(go.Scatter(x=df.index, y=rolling_mean, mode='lines', name='Rolling Mean'))
    fig2.add_trace(go.Scatter(x=df.index, y=rolling_std, mode='lines', name='Rolling Std'))
    fig2.update_layout(
        title='Rolling Mean & Std',
        xaxis_title='Date',
        yaxis_title='Close',
        height=350,
        xaxis=dict(rangeslider=dict(visible=True), type="date")
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 3. Decomposition (Trend, Seasonality, Residuals) - Interactive with rangeslider
    period = min(252, max(2, len(df) // 2))
    decomposition = seasonal_decompose(df['Close'], model='additive', period=period)
    fig3 = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=['Trend', 'Seasonality', 'Residuals'])
    fig3.add_trace(go.Scatter(x=df.index, y=decomposition.trend, name='Trend'), row=1, col=1)
    fig3.add_trace(go.Scatter(x=df.index, y=decomposition.seasonal, name='Seasonality'), row=2, col=1)
    fig3.add_trace(go.Scatter(x=df.index, y=decomposition.resid, name='Residuals'), row=3, col=1)
    fig3.update_layout(
        height=600,
        showlegend=False,
        xaxis=dict(rangeslider=dict(visible=True), type="date")
    )
    st.plotly_chart(fig3, use_container_width=True)

    # 4. ACF & PACF (Static, but zoomable in plotly)
    import io
    # ACF
    fig_acf, ax_acf = plt.subplots(figsize=(6, 2.5))
    plot_acf(df['Close'], lags=50, ax=ax_acf)
    ax_acf.set_title('ACF')
    buf_acf = io.BytesIO()
    plt.tight_layout()
    fig_acf.savefig(buf_acf, format="png")
    plt.close(fig_acf)
    st.image(buf_acf, caption="ACF", use_container_width=True)
    # PACF
    fig_pacf, ax_pacf = plt.subplots(figsize=(6, 2.5))
    plot_pacf(df['Close'], lags=50, ax=ax_pacf)
    ax_pacf.set_title('PACF')
    buf_pacf = io.BytesIO()
    plt.tight_layout()
    fig_pacf.savefig(buf_pacf, format="png")
    plt.close(fig_pacf)
    st.image(buf_pacf, caption="PACF", use_container_width=True)

def plot_eda_static(df, ticker, start_date, end_date):
    # Filter by selected date range
    df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

    st.subheader(f"{ticker} - EDA Plots")

    # Only plot if enough data
    if len(df) < 30:
        st.warning("Not enough data for full EDA plots. Showing only Close Price.")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(df['Close'])
        ax.set_title('Close Price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Close')
        st.pyplot(fig)
        return

    # 1. Close Price
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df['Close'])
    ax.set_title('Close Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close')
    st.pyplot(fig)

    # 2. Rolling Mean
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df['Close'], label='Original')
    ax.plot(df['Close'].rolling(30).mean(), label='Rolling Mean')
    ax.set_title('Rolling Mean')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close')
    ax.legend()
    st.pyplot(fig)

    # 3. Rolling Std
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df['Close'], label='Original')
    ax.plot(df['Close'].rolling(30).std(), label='Rolling Std')
    ax.set_title('Rolling Std')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close')
    ax.legend()
    st.pyplot(fig)

    # 4. Decomposition - Trend
    period = min(252, max(2, len(df) // 2))
    decomposition = seasonal_decompose(df['Close'], model='additive', period=period)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(decomposition.trend)
    ax.set_title('Trend')
    ax.set_xlabel('Date')
    ax.set_ylabel('Trend')
    st.pyplot(fig)

    # 5. Decomposition - Seasonality
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(decomposition.seasonal)
    ax.set_title('Seasonality')
    ax.set_xlabel('Date')
    ax.set_ylabel('Seasonality')
    st.pyplot(fig)

    # 6. Decomposition - Residuals
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(decomposition.resid)
    ax.set_title('Residuals')
    ax.set_xlabel('Date')
    ax.set_ylabel('Residuals')
    st.pyplot(fig)

# Add a dictionary for stock descriptions
STOCK_DESCRIPTIONS = {
    "DIVISLAB.NS": "Divislabs laboratories ltd",
    "INDIGO.NS": "InterGlobe Aviation Limited (IndiGo) is India's largest passenger airline with a market share of 57.7% as of January 2023.",
    "TCS.NS": "Tata Consultancy Services Limited (TCS) is an Indian multinational information technology (IT) services and consulting company.",
    "RELIANCE.NS": "Reliance Industries Limited is an Indian multinational conglomerate company.",
    "HDFCBANK.NS": "HDFC Bank Limited is an Indian banking and financial services company.",
    "ICICIBANK.NS": "ICICI Bank Limited is an Indian multinational bank and financial services company.",
    "INFY.NS": "Infosys Limited is an Indian multinational information technology company.",
    "HINDUNILVR.NS": "Hindustan Unilever Limited is an Indian consumer goods company.",
    "ITC.NS": "ITC Limited is an Indian conglomerate company.",
    "LT.NS": "Larsen & Toubro Limited is an Indian multinational conglomerate.",
    "SBIN.NS": "State Bank of India is an Indian multinational, public sector banking and financial services company.",
    "KOTAKBANK.NS": "Kotak Mahindra Bank Limited is an Indian banking and financial services company.",
    "BHARTIARTL.NS": "Bharti Airtel Limited is an Indian multinational telecommunications services company.",
    "BAJFINANCE.NS": "Bajaj Finance Limited is an Indian non-banking financial company.",
    "ASIANPAINT.NS": "Asian Paints Limited is an Indian multinational paint company.",
    "AXISBANK.NS": "Axis Bank Limited is an Indian banking and financial services company.",
    "HCLTECH.NS": "HCL Technologies Limited is an Indian multinational IT services and consulting company.",
    "MARUTI.NS": "Maruti Suzuki India Limited is an Indian automobile manufacturer.",
    "SUNPHARMA.NS": "Sun Pharmaceutical Industries Limited is an Indian multinational pharmaceutical company.",
    "TITAN.NS": "Titan Company Limited is an Indian luxury products company.",
    "ULTRACEMCO.NS": "UltraTech Cement Limited is an Indian cement company.",
    "TATAMOTORS.NS": "Tata Motors Limited is an Indian multinational automotive manufacturing company.",
    "WIPRO.NS": "Wipro Limited is an Indian multinational corporation that provides IT, consulting and business process services.",
    "ADANIGREEN.NS": "No description available.",
    "ADANIPORTS.NS": "No description available.",
    "ADANIPOWER.NS": "No description available.",
    "ADANITRANS.NS": "No description available.",
    "AMBUJACEM.NS": "No description available.",
    "APOLLOHOSP.NS": "No description available.",
    "APOLLOTYRE.NS": "No description available.",
    "ASHOKLEY.NS": "No description available.",
    "AUROPHARMA.NS": "No description available.",
    "BANDHANBNK.NS": "No description available.",
    "BANKBARODA.NS": "No description available.",
    "BERGEPAINT.NS": "No description available.",
    "BEL.NS": "No description available.",
    "BHEL.NS": "No description available.",
    "BIOCON.NS": "No description available.",
    "BOSCHLTD.NS": "No description available.",
    "BPCL.NS": "No description available.",
    "BRITANNIA.NS": "No description available.",
    "CANBK.NS": "No description available.",
    "CHOLAFIN.NS": "No description available.",
    "CIPLA.NS": "No description available.",
    "COALINDIA.NS": "No description available.",
    "COLPAL.NS": "No description available.",
    "CONCOR.NS": "No description available.",
    "COROMANDEL.NS": "No description available.",
    "CROMPTON.NS": "No description available.",
    "CUMMINSIND.NS": "No description available.",
    "DABUR.NS": "No description available.",
    "DALBHARAT.NS": "No description available.",
    "DEEPAKNTR.NS": "No description available.",
    "DLF.NS": "No description available.",
    "DIXON.NS": "No description available.",
    "DRREDDY.NS": "No description available.",
    "EICHERMOT.NS": "No description available.",
    "ESCORTS.NS": "No description available.",
    "FEDERALBNK.NS": "No description available.",
    "GAIL.NS": "No description available.",
    "GLAND.NS": "No description available.",
    "GMRINFRA.NS": "No description available.",
    "GODREJCP.NS": "No description available.",
    "GODREJPROP.NS": "No description available.",
    "GRASIM.NS": "No description available.",
    "GUJGASLTD.NS": "No description available.",
    "HAVELLS.NS": "No description available.",
    "HDFCLIFE.NS": "No description available.",
    "HEROMOTOCO.NS": "No description available.",
    "HINDALCO.NS": "No description available.",
    "HINDPETRO.NS": "No description available.",
    "HINDZINC.NS": "No description available.",
    "ICICIGI.NS": "No description available.",
    "ICICIPRULI.NS": "No description available.",
    "IDFCFIRSTB.NS": "No description available.",
    "IGL.NS": "No description available.",
    "INDHOTEL.NS": "No description available.",
    "INDIAMART.NS": "No description available.",
    "INDIGO.NS": "No description available.",
    "INDUSINDBK.NS": "No description available.",
    "INDUSTOWER.NS": "No description available.",
    "IOC.NS": "No description available.",
    "IRCTC.NS": "No description available.",
    "ITC.NS": "No description available.",
    "JINDALSTEL.NS": "No description available.",
    "JSWSTEEL.NS": "No description available.",
    "JUBLFOOD.NS": "No description available.",
    "KOTAKBANK.NS": "No description available.",
    "LALPATHLAB.NS": "No description available.",
    "LICHSGFIN.NS": "No description available.",
    "LTIM.NS": "No description available.",
    "LTTS.NS": "No description available.",
    "LUPIN.NS": "No description available.",
    "M&M.NS": "No description available.",
    "M&MFIN.NS": "No description available.",
    "MANAPPURAM.NS": "No description available.",
    "MARICO.NS": "No description available.",
    "MARUTI.NS": "No description available.",
    "MCDOWELL-N.NS": "No description available.",
    "MCX.NS": "No description available.",
    "METROPOLIS.NS": "No description available.",
    "MFSL.NS": "No description available.",
    "MGL.NS": "No description available.",
    "MOTHERSON.NS": "No description available.",
    "MPHASIS.NS": "No description available.",
    "MRF.NS": "No description available.",
    "MUTHOOTFIN.NS": "No description available.",
    "NAM-INDIA.NS": "No description available.",
    "NAUKRI.NS": "No description available.",
    "NAVINFLUOR.NS": "No description available.",
    "NESTLEIND.NS": "No description available.",
    "NMDC.NS": "No description available.",
    "NTPC.NS": "No description available.",
    "OBEROIRLTY.NS": "No description available.",
    "OFSS.NS": "No description available.",
    "ONGC.NS": "No description available.",
    "PAGEIND.NS": "No description available.",
    "PEL.NS": "No description available.",
    "PETRONET.NS": "No description available.",
    "PIIND.NS": "No description available.",
    "PIDILITIND.NS": "No description available.",
    "PNB.NS": "No description available.",
    "POLYCAB.NS": "No description available.",
    "POWERGRID.NS": "No description available.",
    "PVRINOX.NS": "No description available.",
    "RAMCOCEM.NS": "No description available.",
    "RBLBANK.NS": "No description available.",
    "RECLTD.NS": "No description available.",
    "RELIANCE.NS": "No description available.",
    "SAIL.NS": "No description available.",
    "SBICARD.NS": "No description available.",
    "SBILIFE.NS": "No description available.",
    "SHREECEM.NS": "No description available.",
    "SIEMENS.NS": "No description available.",
    "SRF.NS": "No description available.",
    "SRTRANSFIN.NS": "No description available.",
    "SUNPHARMA.NS": "No description available.",
    "SUNTV.NS": "No description available.",
    "SYNGENE.NS": "No description available.",
    "TATACHEM.NS": "No description available.",
    "TATACOMM.NS": "No description available.",
    "TATACONSUM.NS": "No description available.",
    "TATAMOTORS.NS": "No description available.",
    "TATAPOWER.NS": "No description available.",
    "TATASTEEL.NS": "No description available.",
    "TCS.NS": "No description available.",
    "TECHM.NS": "No description available.",
    "TITAN.NS": "No description available.",
    "TORNTPHARM.NS": "No description available.",
    "TORNTPOWER.NS": "No description available.",
    "TRENT.NS": "No description available.",
    "TVSMOTOR.NS": "No description available.",
    "UBL.NS": "No description available.",
    "ULTRACEMCO.NS": "No description available.",
    "UNIONBANK.NS": "No description available.",
    "UPL.NS": "No description available.",
    "VBL.NS": "No description available.",
    "VEDL.NS": "No description available.",
    "VOLTAS.NS": "No description available.",
    "WHIRLPOOL.NS": "No description available.",
    "WIPRO.NS": "No description available.",
    "ZEEL.NS": "No description available.",
    "ZYDUSLIFE.NS": "No description available.",
    "OLAELCTRIC.NS": "Ola Electric Mobility Limited is an Indian electric vehicle manufacturer, primarily known for its electric scooters.",
    # ...add or update descriptions as needed for all 200 stocks...
}

# List of all NS200 stock symbols for dropdown
NS200_SYMBOLS = list(STOCK_DESCRIPTIONS.keys())

def get_stock_description(ticker):
    return STOCK_DESCRIPTIONS.get(
        ticker.upper(),
        "No description available for this stock symbol."
    )

def get_stock_trend_summary(df):
    """Return a simple summary of the trend for the selected date range.
    Uses both numerical and OpenCV-based visual trend detection."""
    close = df['Close'].dropna()
    if len(close) < 2:
        return "Not enough data to determine trend."
    # Numerical trend
    pct_change = float((close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100)
    if pct_change > 5:
        trend_text = f"The stock has an upward trend (+{pct_change:.2f}%) over the selected period."
    elif pct_change < -5:
        trend_text = f"The stock has a downward trend ({pct_change:.2f}%) over the selected period."
    else:
        trend_text = f"The stock has been relatively stable ({pct_change:.2f}%) over the selected period."
    # OpenCV-based trend detection (using linear fit)
    y = close.values.astype(np.float32)
    x = np.arange(len(y)).astype(np.float32)
    if len(y) > 1:
        # Fit a line using OpenCV
        [vx, vy, x0, y0] = cv2.fitLine(np.column_stack((x, y)), cv2.DIST_L2, 0, 0.01, 0.01)
        slope = float(vy / vx) if vx != 0 else 0
        if slope > 0.01:
            cv_trend = "OpenCV also detects an upward trend."
        elif slope < -0.01:
            cv_trend = "OpenCV also detects a downward trend."
        else:
            cv_trend = "OpenCV detects a stable trend."
    else:
        cv_trend = "OpenCV: Not enough data for trend detection."
    return f"{trend_text}\n{cv_trend}"

def get_nifty200_tickers():
    """
    Scrape Nifty 200 tickers from NSE India website.
    Returns a list of ticker symbols (e.g., ['DIVISLAB.NS', ...])
    """
    url = "https://www1.nseindia.com/content/indices/ind_nifty200list.csv"
    try:
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(pd.compat.StringIO(response.text))
        tickers = [row['Symbol'] + ".NS" for _, row in df.iterrows()]
        return tickers
    except Exception as e:
        print(f"Error fetching Nifty 200 tickers: {e}")
        return []

# Utility: Print all yfinance tickers matching a substring (case-insensitive)
def print_matching_yfinance_tickers(substring):
    """
    Print all NS200 yfinance tickers whose symbol or description contains the given substring (case-insensitive).
    """
    matches = []
    for symbol, desc in STOCK_DESCRIPTIONS.items():
        if substring.lower() in symbol.lower() or substring.lower() in desc.lower():
            matches.append((symbol, desc))
    for symbol, desc in matches:
        print(f"{symbol}: {desc}")

# Example usage (uncomment to use in script or interactive session):
# print_matching_yfinance_tickers('divi')

if __name__ == "__main__":
    st.title("Stock EDA Dashboard")
    # Data source dropdown (remove investpy option)
    data_source = st.selectbox("Select Data Source", options=["yfinance"], index=0)
    ticker = st.selectbox("Select NS200 Stock Symbol", options=NS200_SYMBOLS, index=0)
    if ticker:
        st.info(get_stock_description(ticker))
        import datetime
        end = pd.Timestamp.today()
        start = end - pd.DateOffset(years=1)
        if data_source == "yfinance":
            df = download_data(ticker, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
        else:
            df = pd.DataFrame()
        if not df.empty:
            df = preprocess_data(df)
            min_date = df.index.min().to_pydatetime()
            max_date = df.index.max().to_pydatetime()
            date_range = st.slider(
                "Select date range:",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                format="YYYY-MM-DD"
            )
            st.subheader("Ask about the trend of this stock")
            trend_query = st.text_input("Ask a question about the trend (e.g., 'What is the trend?')", key="trend_query")
            df_selected = df[(df.index >= pd.Timestamp(date_range[0])) & (df.index <= pd.Timestamp(date_range[1]))]
            if trend_query:
                st.success(get_stock_trend_summary(df_selected))
            plot_eda_static(df, ticker, pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))