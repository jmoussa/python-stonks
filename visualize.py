import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

style.use('ggplot')


def visualize_data():
    df = pd.read_csv('sp500_joined_closes.csv', parse_dates=True)
    # df['AAPL'].plot()
    # plt.show()
    # Correlation table df
    df_corr = df.corr()
    data = df_corr.values

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # red yellow green heatmap
    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)

    # setup the ticks at halfway tickmarks
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1, 1)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    visualize_data()
