import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as web
from datetime import datetime
from sklearn import preprocessing
import numpy as np
from finta import TA
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
from keras.callbacks import History 
import numpy as np
import matplotlib.gridspec as grd
import matplotlib.dates as mdates
import matplotlib.ticker as ticker




def train_test_split_preparation(new_df, data_set_points, train_split):
    new_df = new_df.loc[1:]

    #Preparation of train test set.
    train_indices = int(new_df.shape[0] * train_split)

    train_data = new_df[:train_indices]
    test_data = new_df[train_indices:]
    test_data = test_data.reset_index()
    test_data = test_data.drop(columns = ['index'])
    
    train_arr = np.diff(train_data.loc[:, ['Adj Close']].values, axis = 0)
    test_arr = np.diff(test_data.loc[:, ['Adj Close']].values, axis = 0)


    X_train = np.array([train_arr[i : i + data_set_points] for i in range(len(train_arr) - data_set_points)])


    y_train = np.array([train_arr[i + data_set_points] for i in range(len(train_arr) - data_set_points)])

    X_test = np.array([test_arr[i : i + data_set_points] for i in range(len(test_arr) - data_set_points)])

    y_test = np.array([test_data['Adj Close'][i + data_set_points] for i in range(len(test_arr) - data_set_points)])


    return X_train, y_train, X_test, y_test, test_data

def lstm_model(X_train, y_train, data_set_points):
    #Setting of seed
    tf.random.set_seed(20)
    np.random.seed(10)

    lstm_input = Input(shape=(data_set_points, 1), name='input_for_lstm')

    inputs = LSTM(21, name='first_layer', return_sequences = True)(lstm_input)

    inputs = Dropout(0.1, name='first_dropout_layer')(inputs)
    inputs = LSTM(32, name='lstm_1')(inputs)
    inputs = Dropout(0.05, name='lstm_dropout_1')(inputs)
    inputs = Dense(32, name='first_dense_layer')(inputs)
    inputs = Dense(1, name='dense_layer')(inputs)
    output = Activation('linear', name='output')(inputs)

    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam(lr = 0.002)

    model.compile(optimizer=adam, loss='mse')
    models = model.fit(x=X_train, y=y_train, batch_size=15, epochs=25, shuffle=True, validation_split = 0.1)

    return model

def buy_sell_trades(actual, predicted):
    pred_df = pd.DataFrame()
    pred_df['Predictions'] = predicted


    y_pct_change = pred_df.pct_change()

    money = 10000
    number_of_stocks = (int)(10000 / actual[0])
    left = 10000 - (int)(10000 / actual[0]) * actual[0] + actual[len(actual) - 1] * number_of_stocks

    number_of_stocks = 0

    buying_percentage_threshold = 0.0015 #as long as we have a 0.15% increase/decrease we buy/sell the stock
    selling_percentage_threshold = 0.0015

    for i in range(len(actual) - 1):    
        if y_pct_change['Predictions'][i + 1] > buying_percentage_threshold:
            for j in range(100, 0, -1):
                if (money >= j * actual[i]):
                    money -= j * actual[i]
                    number_of_stocks += j
                    break
        elif  y_pct_change['Predictions'][i + 1] < -selling_percentage_threshold:
            for j in range(100, 0, -1):
                if (number_of_stocks >= j):
                    money += j * actual[i]
                    number_of_stocks -= j
                    break

    money += number_of_stocks * actual[len(actual) - 1]

    print(money)
    print(left)

    return y_pct_change


def plot_buy_sell_trades(y_pct_change, actual):
    plt.figure()
    gs = grd.GridSpec(2, 1, height_ratios=[7,5], wspace=0.1)

    plt.gcf().set_size_inches(16, 10, forward=True)


    plt.subplot(gs[1])

    plt.plot(y_pct_change)
    # p = ax.imshow(y_pct_change,interpolation='nearest',aspect='auto') # set the aspect ratio to auto to fill the space. 
    plt.xlabel('Day')
    plt.ylabel('Percentage change')
    # plt.xlim(1,140)
    # lists = [False, True, True]
    # fig, ax1 = plt.subplots()
    # ax.plot([1,2, 3], lists,'rv', size = 400)

    days_x = np.array([x for x in range(len(y_pct_change))])

    mask1 = y_pct_change > 0.001
    mask2 = y_pct_change < -0.003
    mask1 = mask1.to_numpy()
    mask1 = mask1.flatten()
    mask2 = mask2.to_numpy()
    mask2 = mask2.flatten()

    # color red/green bar in it's own axis
    plt.subplot(gs[0])
    plt.bar(days_x[mask1], actual[mask1], color = "green")
    plt.bar(days_x[mask2], actual[mask2], color = "red")
    plt.xlabel('Day')
    plt.ylabel('Price of Google stock')

    plt.show()





if __name__ == "__main__":
    start_date = datetime(2010, 9, 1)
    end_date = datetime(2020, 8, 31)
    

    #invoke to_csv for df dataframe object from 
    #DataReader method in the pandas_datareader library
    # df = web.DataReader("GOOGL", 'yahoo', start_date, end_date)
    
    
    # df.to_csv('./stock-project/google.csv')

    #pulling of google data from csv file
    stock_df = pd.read_csv('./stock-project/csv_files/google_stocks_data.csv') #Note this data was pulled on 6 October 2020, some data may have changed since then 


    train_split = 0.7
    
    data_set_points = 21

    new_df = stock_df[['Adj Close']].copy()

    X_train, y_train, X_test, y_test, test_data = train_test_split_preparation(new_df, data_set_points, train_split)

    
    dataset = []
    test = []
    

    model = lstm_model(X_train, y_train, data_set_points)

    y_pred = model.predict(X_test)

    print(y_pred)

    y_pred = y_pred.flatten()


    actual = np.array([test_data['Adj Close'][i + data_set_points].copy() for i in range(len(test_data) - data_set_points)])

    reference = test_data['Adj Close'][data_set_points - 1]

    predicted = []

    predicted.append(reference)

    for i in y_pred:
        reference += i
        predicted.append(reference)

    predicted = np.array(predicted)

    print(predicted)

    real = plt.plot(actual, label='Actual Price')
    pred = plt.plot(predicted, label='Predicted Price')

    plt.legend(['Actual Price', 'Predicted Price'])
    plt.gcf().set_size_inches(15, 10, forward=True)
    
    plt.show()

    y_pct_change = buy_sell_trades(actual, predicted)

    plot_buy_sell_trades(y_pct_change, actual)




    
   
