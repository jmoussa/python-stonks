from utils import download_ticker, plot_ma  # plot_candlestick
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--ticker", help="stock ticker that you would like to generate a chart for")

    args = parser.parse_args()

    if args.ticker is None:
        parser.print_help(sys.stderr)

    return args


def main():
    args = parse_args()
    user_entered_ticker = args.ticker
    df, ticker = download_ticker(ticker=user_entered_ticker)
    plot_ma(df, ticker)
    # plot_candlestick(df, ticker)


if __name__ == "__main__":
    main()
