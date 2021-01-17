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
np.random.seed(20)
tf.random.set_seed(10)

def on_balance_volume_creation(stock_df):
    # Adding of on balance volume to dataframe
    
    new_df = pd.DataFrame({})

    new_df = stock_df[['Adj Close']].copy()


    new_balance_volume = [0]
    tally = 0

    for i in range(1, len(new_df)):
        if (stock_df['Adj Close'][i] > stock_df['Adj Close'][i - 1]):
            tally += stock_df['Volume'][i]
        elif (stock_df['Adj Close'][i] < stock_df['Adj Close'][i - 1]):
            tally -= stock_df['Volume'][i]
        new_balance_volume.append(tally)

    new_df['On_Balance_Volume'] = new_balance_volume
    minimum = min(new_df['On_Balance_Volume'])

    new_df['On_Balance_Volume'] = new_df['On_Balance_Volume'] - minimum
    new_df['On_Balance_Volume'] = (new_df['On_Balance_Volume']+1).transform(np.log)

    return new_df

def add_technical_indicators(new_df):
    # Adding of technical indicators to data frame (Exponential moving average and Bollinger Band)
    edited_df = pd.DataFrame()

    edited_df['open'] = stock_df['Open']
    edited_df['high'] = stock_df['High']
    edited_df['low'] = stock_df['Low']
    edited_df['close'] = stock_df['Close']
    edited_df['volume'] = stock_df['Volume']
    edited_df.head()

    ema = TA.EMA(edited_df)
    bb = TA.BBANDS(edited_df)

    new_df['Exponential_moving_average'] = ema.copy()

    new_df = pd.concat([new_df, bb], axis = 1)


    for i in range(19):
        new_df['BB_MIDDLE'][i] = new_df.loc[i, 'Exponential_moving_average']
    
        if i != 0:
            higher = new_df.loc[i, 'BB_MIDDLE'] + 2 * new_df['Adj Close'].rolling(i + 1).std()[i]
            lower = new_df.loc[i, 'BB_MIDDLE'] - 2 * new_df['Adj Close'].rolling(i + 1).std()[i]
            new_df['BB_UPPER'][i] = higher
            new_df['BB_LOWER'][i] = lower
        else:
            new_df['BB_UPPER'][i] = new_df.loc[i, 'BB_MIDDLE']
            new_df['BB_LOWER'][i] = new_df.loc[i, 'BB_MIDDLE']
    return new_df


def train_test_split_preparation(new_df, train_split):
    #Preparation of train test set.
    train_indices = int(new_df.shape[0] * train_split)

    train_data = new_df[:train_indices]
    test_data = new_df[train_indices:]

    test_data = test_data.reset_index()
    test_data = test_data.drop(columns = ['index'])

    normaliser = preprocessing.MinMaxScaler()
    train_normalised_data = normaliser.fit_transform(train_data)

    test_normalised_data = normaliser.transform(test_data)

    X_train = np.array([train_normalised_data[:,0:][i : i + history_points].copy() for i in range(len(train_normalised_data) - history_points)])

    y_train = np.array([train_normalised_data[:,0][i + history_points].copy() for i in range(len(train_normalised_data) - history_points)])
    y_train = np.expand_dims(y_train, -1)

    y_normaliser = preprocessing.MinMaxScaler()
    next_day_close_values = np.array([train_data['Adj Close'][i + history_points].copy() for i in range(len(train_data) - history_points)])
    next_day_close_values = np.expand_dims(next_day_close_values, -1)

    y_normaliser.fit(next_day_close_values)

     
    X_test = np.array([test_normalised_data[:,0:][i  : i + history_points].copy() for i in range(len(test_normalised_data) - history_points)])
    

    y_test = np.array([test_data['Adj Close'][i + history_points].copy() for i in range(len(test_data) - history_points)])
    
    y_test = np.expand_dims(y_test, -1)

    return X_train, y_train, X_test, y_test, y_normaliser

def lstm_model(X_train, y_train, history_points):
    lstm_input = Input(shape=(history_points, 6), name='first_lstm')

    inputs = LSTM(63, name='lstm_input')(lstm_input)
    inputs = Dropout(0.3, name='dropout')(inputs)
    inputs = Dense(256, name='first_dense_layer')(inputs)
    inputs = Activation('sigmoid', name='first_sigmoid_layer')(inputs)
    inputs = Dense(1, name='second_dense_layer')(inputs)

    output = Activation('linear', name='linear_output')(inputs)
    model = Model(inputs=lstm_input, outputs=output)

    adam = optimizers.Adam(lr = 0.0035)
    model.compile(optimizer=adam, loss='mse')

    model.fit(x=X_train, y=y_train, batch_size=56, epochs=150, shuffle=True, validation_split = 0.1)

    return model




if __name__ == "__main__":
    start_date = datetime(2010, 9, 1)
    end_date = datetime(2020, 8, 31)
    

    #invoke to_csv for df dataframe object from 
    #DataReader method in the pandas_datareader library
    # df = web.DataReader("GOOGL", 'yahoo', start_date, end_date)
    
    #invoke to_csv for df dataframe object from 
    #DataReader method in the pandas_datareader library
    
    #..\first_yahoo_prices_to_csv_demo.csv must not
    #be open in another app, such as Excel
    
    # df.to_csv('google_stocks_data.csv')

    #pulling of google data from csv file
    stock_df = pd.read_csv('./stock-project/google_stocks_data.csv')   #Note this data was pulled on 6 October 2020, some data may have changed since then 

    train_split = 0.6
    
    history_points = 63
    
    new_df = on_balance_volume_creation(stock_df)

    new_df = add_technical_indicators(new_df)

    X_train, y_train, X_test, y_test, y_reverse_normaliser = train_test_split_preparation(new_df, train_split)

    model = lstm_model(X_train, y_train, history_points)

    y_pred = model.predict(X_test)
    y_pred = y_reverse_normaliser.inverse_transform(y_pred)

    real = plt.plot(y_test, label='Actual Price')
    pred = plt.plot(y_pred, label='Predicted Price')

    plt.gcf().set_size_inches(16, 9, forward=True)
    plt.title('Plot of real price and predicted price against number of days')
    plt.xlabel('Number of days')
    plt.ylabel('Adjusted Close Price($)')

    plt.legend(['Actual Price', 'Predicted Price'])

 
    plt.show()    
