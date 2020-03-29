import sys
import numpy as np
import pandas as pd

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

days_in_future = 7


# classification
def process_data_for_labels(ticker):
    df = pd.read_csv("./csvs/sp500_joined_closes.csv", index_col=0)
    # get tickers
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, days_in_future + 1):
        df[f"{ticker}_{i}d"] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df


# takes in array of columns (the week of future percent changes)
def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            # buy
            return 1
        elif col < -requirement:
            # sell
            return -1
    # hold
    return 0


def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)

    df[f"{ticker}_target"] = list(
        map(buy_sell_hold, *[df["{}_{}d".format(ticker, i)] for i in range(1, days_in_future + 1, 1)])
    )

    vals = df[f"{ticker}_target"].values.tolist()
    str_vals = [str(i) for i in vals]
    print(f"Data Spread: {Counter(str_vals)}")

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    # feature sets and labels
    X = df_vals.values
    y = df[f"{ticker}_target"].values

    return X, y, df


def do_ml(ticker):
    """
        X is the percent change for all companies
        y is the 1, 0, -1 target evaluation for those companies
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
    print(f"Confidence: {confidence}")

    # Make prediction
    predictions = clf.predict(X_test)

    print(f"Predicted spread: {Counter(predictions)}")
    return confidence


if __name__ == "__main__":
    if sys.argv[1] is not None:
        do_ml(sys.argv[1])
    else:
        print("Please supply a ticker to perform analysis on")
