from datetime import datetime
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

def import_df(file_path):
    return pd.read_csv(
        file_path, 
        parse_dates = DATE_COLS, 
        date_parser = lambda x: datetime.strptime(x, '%Y-%m-%d') # so that Dates col is Timestamp, not str
    )

if __name__ == "__main__":
    DATE_COLS = ['Dates']

    ts_df = import_df('./stock-project/csv_files/predicted_actual_price.csv')
    buy_sell_df = import_df('./stock-project/csv_files/buy_sell_dates.csv')

    buy_dates = buy_sell_df[buy_sell_df.Buy == 1].Dates
    sell_dates = buy_sell_df[buy_sell_df.Sell == 1].Dates

    buy_decisions = ts_df[ts_df.Dates.isin(buy_dates)]
    sell_decisions = ts_df[ts_df.Dates.isin(sell_dates)]

    ts_df['next_day_pct_change_pred'] = ts_df.Predictions.pct_change(periods=1).shift(-1) * 100
    pct_change_df = ts_df.dropna()

    months = mdates.MonthLocator(interval=3)

    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)


    ## FIRST SUBPLOT

    ax1.plot(ts_df.Dates, ts_df.Actual)
    ax1.scatter(buy_decisions.Dates, buy_decisions.Actual * 0.95, color = 'green', marker = '^')
    ax1.scatter(sell_decisions.Dates, sell_decisions.Actual * 1.05, color = 'red', marker = 'v')

    # format title, x/y axis
    dollar_formatter = ticker.FormatStrFormatter('$ %1.2f')
    ax1.yaxis.set_major_formatter(dollar_formatter)
    ax1.xaxis.set_major_locator(months)
    ax1.set_ylabel('Adj Close')
    ax1.set_title('Graph of Buy/Sell Timestamps on Test Set')


    ## SECOND SUBPLOT

    ax2.plot(pct_change_df.Dates, pct_change_df.next_day_pct_change_pred, color='darkblue')
    ax2.axhline(y=0.15, color='green', label='Buy threshold (+0.15%)')
    ax2.axhline(y=-0.15, color='red', label='Sell threshold ( -0.15%)')
    ax2.scatter(buy_decisions.Dates, [0.15] * len(buy_decisions), color='green', label='Buy decisions')
    ax2.scatter(buy_decisions.Dates, [-0.15] * len(sell_decisions), color='red', label='Sell decisions')

    # format title, x/y axis
    ax2.legend()
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax2.xaxis.set_major_locator(months)
    ax2.set_ylabel('Next Day Prediction % Change')
    ax2.set_xlabel('Date')


    ## Output to png and plot in notebook

    plt.savefig('buy-sell-decisions.png',bbox='tight')
    plt.show()





