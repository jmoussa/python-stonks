"""
    Utility functions for plotting and scraping
"""

import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader.data as web
import os

style.use("ggplot")
ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)


def download_ticker(full_path=None, ticker="DIS"):
    print("Ticker: ", ticker)
    start = dt.datetime(2010, 1, 1)
    end = dt.datetime.now()

    df = web.DataReader(ticker, "yahoo", start, end)
    # df = web.get_data_yahoo(ticker, start, end)
    if full_path is None:
        if not os.path.exists("./tmp"):
            os.mkdir("./tmp")
        full_path = os.path.join("./tmp/", f"{ticker}.{start.strftime('%Y%m%d')}.{end.strftime('%Y%m%d')}.csv")

    df.to_csv(full_path)
    return df, ticker


def read_ticker(filename):
    df = pd.read_csv(filename, parse_dates=True, index_col=0)
    print(df.head())


def plot_ma(df, ticker_name):
    df["200 MA"] = df["Adj Close"].rolling(window=200, min_periods=0).mean()
    df["100 MA"] = df["Adj Close"].rolling(window=100, min_periods=0).mean()
    df["20 MA"] = df["Adj Close"].rolling(window=20, min_periods=0).mean()
    df.dropna(inplace=True)

    # print(df.head())

    ax1.plot(df.index, df["Adj Close"])
    ax1.plot(df.index, df["200 MA"], label="200MA")
    ax1.plot(df.index, df["100 MA"], label="100MA")
    ax1.plot(df.index, df["20 MA"], label="20MA")
    ax1.set_title(f"{ticker_name} Moving Averages")

    ax2.bar(df.index, df["Volume"])
    ax1.legend()
    plt.show()


def plot_candlestick(df, ticker_name):
    ax1.xaxis_date()

    # resample to broader data spectrum
    # open, high, low, close of adj close 10 days & total volume every 10 days
    df_ohlc = df["Adj Close"].resample("10D").ohlc()
    df_volume = df["Volume"].resample("10D").sum()

    # convert date from index to column
    df_ohlc.reset_index(inplace=True)
    df_ohlc["Date"] = df_ohlc["Date"].map(mdates.date2num)

    candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup="g")
    ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
    plt.show()
