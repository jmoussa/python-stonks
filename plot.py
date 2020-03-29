from utils import download_ticker, plot_ma, plot_candlestick
import sys


def main():
    user_entered_ticker = sys.argv[1] if sys.argv[1] is not None else "DIS"
    df, ticker = download_ticker(ticker=user_entered_ticker)
    plot_ma(df, ticker)
    plot_candlestick(df, ticker)


if __name__ == "__main__":
    main()
