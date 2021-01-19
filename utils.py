"""
    Utility functions for plotting and scraping
"""

import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
import logging

logger = logging.getLogger(__name__)

style.use("ggplot")
ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)


def download_ticker(full_path=None, ticker="DIS"):
    print("Ticker: ", ticker)
    start = dt.datetime(2009, 1, 1)
    end = dt.datetime.now()
    try:
        df = web.DataReader(ticker, "yahoo", start, end)
        return df
    except KeyError as e:
        logger.error(f"Key Error: {e} for {ticker}")
        return pd.DataFrame()


def read_ticker(filename):
    df = pd.read_csv(filename, parse_dates=True, index_col=0)
    print(df.head())


def plot_ma(df, ticker_name):
    df["200 MA"] = df["Adj Close"].rolling(window=200, min_periods=0).mean()
    df["100 MA"] = df["Adj Close"].rolling(window=100, min_periods=0).mean()
    df["50 MA"] = df["Adj Close"].rolling(window=50, min_periods=0).mean()
    df.dropna(inplace=True)
    ax1.plot(df.index, df["Adj Close"])
    ax1.plot(df.index, df["200 MA"], label="200MA")
    ax1.plot(df.index, df["100 MA"], label="100MA")
    ax1.plot(df.index, df["50 MA"], label="50MA")
    ax1.set_title(f"{ticker_name} Moving Averages")

    ax2.bar(df.index, df["Volume"])
    ax1.legend()
    plt.show()
