import sys
import logging
import numpy as np
import pandas as pd
import argparse
import coloredlogs

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

days_in_future = 14
percent_change = 0.05
moving_avg_days = 100

coloredlogs.install(level="DEBUG", fmt="%(asctime)s %(hostname)s %(name)s %(message)s")
FORMAT = "%(asctime)s - %(levelname)s: %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


# classification
def process_data_for_labels(ticker):
    df = pd.read_csv("./csvs/sp500_joined_closes.csv", index_col=0)
    # get tickers
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    # add a column for the number of days_in_future you want to calculate
    for i in range(1, days_in_future + 1):
        df[f"{ticker}_{i}d"] = df[ticker].shift(-i)

    df.fillna(0, inplace=True)
    return tickers, df


# takes in array of columns (the week of future percent changes)
def buy_sell_hold(moving_average, *args):
    cols = [c for c in args]

    for col in cols:
        buy_requirement = (percent_change * moving_average) + moving_average  # +2% above moving average
        sell_requirement = moving_average - (percent_change * moving_average)  # -2% below moving average
        if col > buy_requirement:
            # buy
            return 1
        elif col < sell_requirement:
            # sell
            return -1
    # hold
    return 0


def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)

    df[f"{ticker}_{moving_avg_days}MA"] = df[ticker].rolling(window=moving_avg_days, min_periods=0).mean()
    df[f"{ticker}_target"] = list(
        map(buy_sell_hold, df[f"{ticker}_100MA"], *[df[f"{ticker}_{i}d"] for i in range(1, days_in_future + 1, 1)])
    )

    vals = df[f"{ticker}_target"].values.tolist()
    str_vals = [str(i) for i in vals]
    logger.info(f"Data Spread: {Counter(str_vals)}")

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    # actual values for
    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    # feature sets and labels
    X = df_vals.values
    y = df[f"{ticker}_target"].values

    return X, y, df


def do_ml(ticker, test_ticker=None):
    """
        X is the percent change throughout the days for the company
        y is the 1, 0, -1 target evaluation for the company
    """
    X, y, df = extract_featuresets(ticker)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # vote on classifier
    # clf = neighbors.KNeighborsClassifier()
    clf = VotingClassifier(
        [
            ("lsvc", svm.LinearSVC(max_iter=5000, dual=False)),
            ("knn", neighbors.KNeighborsClassifier()),
            ("rfor", RandomForestClassifier()),
        ]
    )

    # fit classifier to training
    clf.fit(X_train, y_train)

    # check how well it fits
    confidence = clf.score(X_test, y_test)
    logger.info(f"{ticker} Trained Confidence: {confidence}")

    # Make prediction
    predictions = clf.predict(X_test)
    logger.info(f"{ticker} Predicted spread: {Counter(predictions)}")

    if test_ticker is not None:
        logger.info(f"TESTING on {test_ticker}")
        X, y, df = extract_featuresets(test_ticker)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9)

        confidence = clf.score(X_test, y_test)
        logger.info(f"{test_ticker} Testing Confidence: {confidence}")

        # Make prediction
        predictions = clf.predict(X_test)
        logger.info(f"{test_ticker} Predicted spread: {Counter(predictions)}")
        return len(predictions)
    else:
        return len(predictions)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", "--train-ticker", help="Name of ticker to get training data for")
    parser.add_argument("-test", "--test-ticker", help="Optional ticker to get data to test on")
    args = parser.parse_args()

    if args.train_ticker is None:
        raise parser.logger.info_help(sys.stderr)

    return args


if __name__ == "__main__":
    args = parse_arguments()
    prediction_count = do_ml(args.train_ticker, args.test_ticker)
    logger.info(f"Prediction Count: {prediction_count}")
