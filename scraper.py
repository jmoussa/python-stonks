import os
import pickle
import requests
import bs4 as bs
import pandas as pd

from utils import download_ticker


def get_data_and_compile(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)

    if not os.path.exists("stock_dfs"):
        os.mkdir("stock_dfs")

    for ticker in tickers:
        # don't forget to trim the stupid freakin '\n'!!!!!
        # otherwise yahoo shits itself
        try:
            ticker = ticker[:-1]
            path = f"stock_dfs/{ticker}.csv"
            if not os.path.exists(path):
                download_ticker(path, ticker)
        except KeyError:
            print("KeyError")

    compile_data()


def save_sp500_tickers():
    resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", {"id": "constituents"})
    tickers = []
    for row in table.findAll("tr")[1:]:
        ticker = row.findAll("td")[0].text
        tickers.append(ticker)

    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)

    return tickers


def compile_data():
    tickers = [f.split(".")[0] for f in os.listdir("stock_dfs") if os.path.isfile(os.path.join("stock_dfs", f))]

    """
    with open('sp500tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)
    """

    main_df = pd.DataFrame()

    for idx, ticker in enumerate(tickers):
        df = pd.read_csv(f"stock_dfs/{ticker}.csv")
        df.set_index("Date", inplace=True)
        df.rename(columns={"Adj Close": ticker}, inplace=True)
        df.drop(["Open", "High", "Low", "Close", "Volume"], 1, inplace=True)

        # Join the dataframes
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how="outer")

        if idx % 10 == 0:
            print(idx)

    # save as csv
    print(main_df.head())
    main_df.to_csv("./csvs/sp500_joined_closes.csv")


if __name__ == "__main__":
    get_data_and_compile(reload_sp500=True)
    # compile_data()
