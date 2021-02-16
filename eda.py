import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as web
from datetime import datetime
import plotly.express as px
import seaborn as sns

def read_data_frame(stock_df):
    print(stock_df.head())
    print(stock_df.describe())


def plot_visualisation(stock_df):
    #Scatterplot of adjusted close price
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])

    fig = px.scatter(stock_df, x = 'Date', y = 'Adj Close')


    fig.show()

    #Plotting correlation plot

    float_df = stock_df.select_dtypes(exclude = "object")
    corr = float_df.corr()
    ax = plt.axes()
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True), ax = ax)

    ax.set_title("Correlation of various features")
    plt.show()



if __name__ == "__main__":
    start_date = datetime(2010, 9, 1)
    end_date = datetime(2020, 8, 31)

    #invoke to_csv for df dataframe object from 
    #DataReader method in the pandas_datareader library
    df = web.DataReader("FB", 'yahoo', start_date, end_date)

    
    # df.to_csv('google_stocks_data.csv')

    #pulling of google data from csv file
    stock_df = pd.read_csv('./stock-project/csv_files/google_stocks_data.csv')

    read_data_frame(stock_df)

    plot_visualisation(stock_df)
